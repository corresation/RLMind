import copy
import math
from dataclasses import asdict, dataclass, field

import torch
from torch import nn

actionDisabledLogit = -1e10
actionMinProb = 1e-11

activations = {"relu": nn.ReLU, "leakyRelu": nn.LeakyReLU, "sigmoid": nn.Sigmoid, "tanh": nn.Tanh, "gelu": nn.GELU}
optims = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW, "adagrad": torch.optim.Adagrad, "rmsprop": torch.optim.RMSprop}


@dataclass
class ModelConfig:
    layerSizes: list = field(default_factory=lambda: [256, 256, 256])
    activation: str = "relu"
    optim: str = "adam"
    layerNorm: bool = True


@dataclass
class PPOConfig:
    timestepsPerIteration: int = 50_000
    batchSize: int = 50_000
    miniBatchSize: int = 0  # 0 = batchSize

    maxEpisodeDuration: float = 120  # Seconds

    # Always pick the most likely action, good for watching, horrible for learning
    deterministic: bool = False

    policy: ModelConfig = field(default_factory=ModelConfig)
    critic: ModelConfig = field(default_factory=ModelConfig)
    sharedHead: ModelConfig = field(default_factory=lambda: ModelConfig([256]))  # Empty layerSizes = no shared head

    attention: bool = False  # Needs sim.AttentionObs() and sharedHead.layerSizes = []
    attentionBlocks: int = 2
    attentionHeads: int = 4
    attentionSize: int = 128

    epochs: int = 2
    policyLR: float = 3e-4
    criticLR: float = 3e-4

    # Scale of the normalized entropy, so it doesn't need changing when the action count changes
    entropyScale: float = 0.018
    # True: entropy only counts available actions
    maskEntropy: bool = False

    clipRange: float = 0.2
    policyTemperature: float = 1

    gaeLambda: float = 0.95
    gaeGamma: float = 0.99
    rewardClipRange: float = 10  # Clip range for normalized rewards, 0 to disable


def makeModel(numInputs, cfg, numOutputs=None):
    layers = []
    for size in cfg.layerSizes:
        layers.append(nn.Linear(numInputs, size))
        if cfg.layerNorm:
            layers.append(nn.LayerNorm(size))
        layers.append(activations[cfg.activation]())
        numInputs = size
    if numOutputs:
        layers.append(nn.Linear(numInputs, numOutputs))
    return nn.Sequential(*layers)


# Ref from https://github.com/Rolv-Arild/EARL-pytorch
# Cross-attention Is All You Need
class AttentionBlock(nn.Module):
    def __init__(self, size, numHeads, activation):
        super().__init__()
        self.numHeads = numHeads
        self.queryNorm, self.tokenNorm, self.ffNorm = nn.LayerNorm(size), nn.LayerNorm(size), nn.LayerNorm(size)
        self.query = nn.Linear(size, size)
        self.keyValue = nn.Linear(size, size * 2)
        self.out = nn.Linear(size, size)
        self.ff = nn.Sequential(nn.Linear(size, size * 4), activations[activation](), nn.Linear(size * 4, size))

        # Starts as identity
        for layer in (self.out, self.ff[-1]):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x, tokens, empty):
        # PyTorch kernels were 4x slower, so here's it written out
        q = self.query(self.queryNorm(x)).unflatten(-1, (self.numHeads, -1))
        k, v = self.keyValue(self.tokenNorm(tokens)).unflatten(-1, (2, self.numHeads, -1)).unbind(2)
        scores = torch.einsum("bhd,bnhd->bhn", q, k) / math.sqrt(q.shape[-1])
        weights = scores.masked_fill(empty[:, None], float("-inf")).softmax(-1)
        x = x + self.out(torch.einsum("bhn,bnhd->bhd", weights, v).flatten(1))
        return x + self.ff(self.ffNorm(x))


class AttentionEncoder(nn.Module):
    def __init__(self, querySize, tokenSize, size, numHeads, numBlocks, activation):
        super().__init__()
        assert numBlocks > 0, "attentionBlocks must be at least 1"
        assert numHeads > 0 and size % numHeads == 0, "attentionSize must be a multiple of attentionHeads"
        self.querySize, self.tokenSize = querySize, tokenSize
        self.queryEmbed = nn.Sequential(nn.Linear(querySize, size), activations[activation](), nn.Linear(size, size))
        self.tokenEmbed = nn.Sequential(nn.Linear(tokenSize, size), activations[activation](), nn.Linear(size, size))
        self.blocks = nn.ModuleList(AttentionBlock(size, numHeads, activation) for _ in range(numBlocks))
        self.norm = nn.LayerNorm(size)

    def forward(self, obs):
        tokens = obs[:, self.querySize:].unflatten(1, (-1, self.tokenSize))
        empty = (tokens == 0).all(-1)
        empty[:, 0] = False  # Ball

        x = self.queryEmbed(obs[:, :self.querySize])
        tokens = self.tokenEmbed(tokens)
        for block in self.blocks:
            x = block(x, tokens, empty)
        return self.norm(x)


def policyProbs(models, obs, masks, temperature):
    if "sharedHead" in models:
        obs = models["sharedHead"](obs)
    logits = models["policy"](obs) / temperature
    probs = torch.softmax(logits + actionDisabledLogit * (~masks), -1)
    return probs.clamp(actionMinProb, 1)


def sampleActions(probs, deterministic):
    if deterministic:
        return probs.argmax(1), None
    actions = torch.multinomial(probs, 1, True)
    logProbs = probs.log().gather(-1, actions)
    return actions.flatten(), logProbs.flatten()


def paramVec(model):
    return torch.cat([p.detach().flatten() for p in model.parameters()])


class PPO:
    def __init__(self, obsSize, numActions, cfg, device, querySize=0, tokenSize=0):
        self.cfg = cfg
        self.device = device
        self.obsSize = obsSize
        self.numActions = numActions
        self.querySize, self.tokenSize = querySize, tokenSize
        if not cfg.miniBatchSize:
            cfg.miniBatchSize = cfg.batchSize
        assert cfg.batchSize % cfg.miniBatchSize == 0, "batchSize must be a multiple of miniBatchSize"

        self.models = {}
        headSize = obsSize
        if cfg.sharedHead.layerSizes:
            self.models["sharedHead"] = makeModel(obsSize, cfg.sharedHead)
            headSize = cfg.sharedHead.layerSizes[-1]

        assert cfg.attention == bool(querySize), "sim.AttentionObs() and attention = True only work together"
        if cfg.attention:
            assert not cfg.sharedHead.layerSizes, "Attention has no shared head, set sharedHead.layerSizes = []"
            for name, numOutputs in (("policy", numActions), ("critic", 1)):
                modelCfg = getattr(cfg, name)
                encoder = AttentionEncoder(
                    querySize, tokenSize, cfg.attentionSize, cfg.attentionHeads, cfg.attentionBlocks, modelCfg.activation)
                self.models[name] = nn.Sequential(encoder, *makeModel(cfg.attentionSize, modelCfg, numOutputs))
        else:
            self.models["policy"] = makeModel(headSize, cfg.policy, numActions)
            self.models["critic"] = makeModel(headSize, cfg.critic, 1)

        modelCfgs = {"sharedHead": cfg.sharedHead, "policy": cfg.policy, "critic": cfg.critic}
        self.optims = {}
        for name, model in self.models.items():
            model.to(device)
            self.optims[name] = optims[modelCfgs[name].optim](model.parameters())
        self.setLearningRates(cfg.policyLR, cfg.criticLR)

        print("Model parameter counts:")
        total = 0
        for name, model in sorted(self.models.items()):
            count = sum(p.numel() for p in model.parameters())
            total += count
            print(f'\t"{name}": {count:,}')
        print(f"\t[Total]: {total:,}")

    def setLearningRates(self, policyLR, criticLR):
        print(f"PPOLearner: Set learning rate to [{policyLR:.6e}, {criticLR:.6e}]")
        lrs = {"policy": policyLR, "critic": criticLR, "sharedHead": min(policyLR, criticLR)}
        for name, optim in self.optims.items():
            for group in optim.param_groups:
                group["lr"] = lrs[name]

    def head(self, obs):
        if "sharedHead" in self.models:
            return self.models["sharedHead"](obs)
        return obs

    def policyProbs(self, obs, masks):
        return policyProbs(self.models, obs, masks, self.cfg.policyTemperature)

    @torch.no_grad()
    def act(self, obs, masks):
        return sampleActions(self.policyProbs(obs, masks), self.cfg.deterministic)

    def copyPolicy(self):
        return {name: copy.deepcopy(model) for name, model in self.models.items() if name != "critic"}

    # Returns act() for numpy obs/masks, as one CUDA graph on GPU (far less launch overhead per step)
    def makeActor(self, numPlayers, obsSize, numActions):
        if self.device.type != "cuda":
            def eagerAct(obs, masks):
                actions, logProbs = self.act(torch.from_numpy(obs), torch.from_numpy(masks).bool())
                return actions.int().numpy(), logProbs.numpy() if logProbs is not None else None
            return eagerAct
        return GraphedActor(self, numPlayers, obsSize, numActions)

    def values(self, obs):
        return self.models["critic"](self.head(obs)).flatten()

    @torch.no_grad()
    def valuesBatched(self, obs):
        if len(obs) == 0:
            return torch.zeros(0, device=self.device)
        return torch.cat([self.values(part) for part in obs.split(self.cfg.miniBatchSize)])

    def entropy(self, probs, masks):
        entropy = -(probs.log() * probs).sum(-1)
        if self.cfg.maskEntropy:
            entropy = entropy / masks.sum(-1).float().log()
        else:
            entropy = entropy / math.log(masks.shape[-1])
        return entropy.mean()

    # exp: dict of device tensors (obs, masks, actions, logProbs, advantages, targetValues)
    def learn(self, exp, report):
        cfg = self.cfg
        before = {name: paramVec(model) for name, model in self.models.items()}
        sums = {"entropy": 0, "kl": 0, "policyLoss": 0, "criticLoss": 0, "clip": 0}
        numMinibatches = 0

        expSize = len(exp["actions"])
        assert expSize >= cfg.batchSize, "Collected less experience than batchSize"
        numBatches = expSize // cfg.batchSize

        for epoch in range(cfg.epochs):
            order = torch.randperm(expSize, device=self.device)
            # The last batch takes the leftover experience
            for batchIdx, batch in enumerate(order[:numBatches * cfg.batchSize].split(cfg.batchSize)):
                if batchIdx == numBatches - 1:
                    batch = order[batchIdx * cfg.batchSize:]

                for mb in batch.split(cfg.miniBatchSize):
                    obs, masks = exp["obs"][mb], exp["masks"][mb]
                    advantages = exp["advantages"][mb]
                    oldLogProbs = exp["logProbs"][mb]
                    sizeRatio = len(mb) / len(batch)

                    probs = self.policyProbs(obs, masks)
                    logProbs = probs.log().gather(-1, exp["actions"][mb].unsqueeze(-1)).flatten()
                    entropy = self.entropy(probs, masks)

                    ratio = torch.exp(logProbs - oldLogProbs)
                    clipped = ratio.clamp(1 - cfg.clipRange, 1 + cfg.clipRange)
                    policyLoss = -torch.min(ratio * advantages, clipped * advantages).mean()
                    criticLoss = nn.functional.mse_loss(self.values(obs), exp["targetValues"][mb])

                    loss = (policyLoss - entropy * cfg.entropyScale + criticLoss) * sizeRatio
                    loss.backward()

                    with torch.no_grad():
                        logRatio = logProbs - oldLogProbs
                        sums["kl"] += ((torch.exp(logRatio) - 1) - logRatio).mean()
                        sums["clip"] += ((ratio - 1).abs() > cfg.clipRange).float().mean()
                        sums["entropy"] += entropy.detach()
                        sums["policyLoss"] += policyLoss.detach()
                        sums["criticLoss"] += criticLoss.detach() * sizeRatio
                    numMinibatches += 1

                for name, model in self.models.items():
                    nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                for optim in self.optims.values():
                    optim.step()
                    optim.zero_grad()

        report["Policy Entropy"] = (sums["entropy"] / numMinibatches).item()
        report["Mean KL Divergence"] = (sums["kl"] / numMinibatches).item()
        report["Policy Loss"] = (sums["policyLoss"] / numMinibatches).item()
        report["Critic Loss"] = (sums["criticLoss"] / numMinibatches).item()
        report["SB3 Clip Fraction"] = (sums["clip"] / numMinibatches).item()
        magnitudeNames = {"policy": "Policy", "critic": "Critic", "sharedHead": "Shared Head"}
        for name, model in self.models.items():
            report[magnitudeNames[name] + " Update Magnitude"] = (paramVec(model) - before[name]).norm().item()

    def attentionConfig(self):
        cfg = self.cfg
        if not cfg.attention:
            return None
        return {"blocks": cfg.attentionBlocks, "heads": cfg.attentionHeads, "size": cfg.attentionSize,
                "querySize": self.querySize, "tokenSize": self.tokenSize}

    def state(self):
        cfg = self.cfg
        return {
            "models": {name: model.state_dict() for name, model in self.models.items()},
            "optims": {name: optim.state_dict() for name, optim in self.optims.items()},
            "configs": {name: asdict(getattr(cfg, name)) for name in self.models},
            "attention": self.attentionConfig(),
            "obsSize": self.obsSize,
            "numActions": self.numActions,
            "policyTemperature": cfg.policyTemperature,
        }

    def loadState(self, state):
        configs = {name: asdict(getattr(self.cfg, name)) for name in self.models}
        attention = self.attentionConfig()
        if state["configs"] != configs or state.get("attention", attention) != attention:
            raise RuntimeError(f"Save has different model settings\nSave: {state['configs']} {state.get('attention')}\nNow: {configs} {attention}")
        for name, model in self.models.items():
            model.load_state_dict(state["models"][name])
        for name, optim in self.optims.items():
            optim.load_state_dict(state["optims"][name])
        self.setLearningRates(self.cfg.policyLR, self.cfg.criticLR)


# Weights are read in place, so optimizer steps and loads show up in the graph without recapturing
class GraphedActor:
    def __init__(self, ppo, numPlayers, obsSize, numActions):
        self.ppo = ppo
        dev = ppo.device
        self.hostObs = torch.zeros(numPlayers, obsSize, pin_memory=True)
        self.hostMasks = torch.zeros(numPlayers, numActions, dtype=torch.uint8, pin_memory=True)
        self.hostActions = torch.zeros(numPlayers, dtype=torch.int32, pin_memory=True)
        self.hostLogProbs = torch.zeros(numPlayers, pin_memory=True)
        self.obs = torch.zeros(numPlayers, obsSize, device=dev)
        self.masks = torch.ones(numPlayers, numActions, dtype=torch.uint8, device=dev)

        # Warm up on a side stream before capturing, as torch asks
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                self.run()
        torch.cuda.current_stream().wait_stream(stream)

        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.actions, self.logProbs = self.run()

    def run(self):
        actions, logProbs = self.ppo.act(self.obs, self.masks.bool())
        return actions.int(), logProbs

    def __call__(self, obs, masks):
        self.hostObs.numpy()[:] = obs
        self.hostMasks.numpy()[:] = masks
        self.obs.copy_(self.hostObs, non_blocking=True)
        self.masks.copy_(self.hostMasks, non_blocking=True)
        self.graph.replay()
        self.hostActions.copy_(self.actions, non_blocking=True)
        if self.logProbs is not None:
            self.hostLogProbs.copy_(self.logProbs, non_blocking=True)
        torch.cuda.current_stream().synchronize()
        return self.hostActions.numpy().copy(), self.hostLogProbs.numpy().copy() if self.logProbs is not None else None
