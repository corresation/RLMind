# RLMind RLBot bot

Runs an RLMind checkpoint in RLBotGUI.

## Step 1: Copy the model

Copy `models.pt` from a checkpoint (`out/build/x64-RelWithDebInfo/checkpoints/<steps>/`) into `model/`:

```
rlbot/
├── model/
│   └── models.pt
├── bot.cfg
├── appearance.cfg
└── bot.py ...
```

The layer sizes are read from `models.pt`, so any RLMind model works. If `models.pt` is missing, the bot won't run.

## Step 2: Configure the bot

### `bot.cfg`
Edit `bot.cfg` to set:
- Bot name
- Developer name
- Description etc.

### `appearance.cfg`
You can edit `appearance.cfg` in two ways:
- Manually in a text editor
- Through **RLBotGUI** (easier)

This file controls:
- Car appearance
- Team colors

Changes apply when the bot is launched.

## Done

1. Open **RLBotGUI**
2. Add `bot.cfg` (or refresh the bot list)
3. Add the bot to a match
4. Launch Rocket League

If the bot doesn't appear or doesn't move, check:
- `models.pt` is in `model/`
- Config edits

## Notes

- RLBotGUI must be installed and working
- All paths in config files are relative to the bot folder
