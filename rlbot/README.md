# RLBot

This runs a trained RLMind checkpoint in Rocket League through RLBot v4. It doesn't train the model or support RLBot v5. No trained model is included.

## Basic setup

1. Install [RLBot v4 / RLBotGUI](https://wiki.rlbot.org/v4/framework/operating-system-support/) and Rocket League on Windows
2. Install this folder's Python packages:

   ```bat
   py -3.11 -m pip install -r rlbot/requirements.txt
   ```

   Run that command from the RLMind root. If RLBotGUI uses its own Python, install these packages there too!

3. Copy `models.pt` and `stats.json` from your preferred checkpoint into `rlbot/bot/`. Without a copied model, the bot uses the newest checkpoint under `out/build/*/checkpoints/`
4. Load `rlbot/bot.cfg` in RLBotGUI, select the bot and start a match

Use the same team size the model was trained for. The supplied setup trains 1v1. The name shown in RLBotGUI is `RLMind`; it comes from `bot.cfg`.
