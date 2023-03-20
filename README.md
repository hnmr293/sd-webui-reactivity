# Reactivity - Checking reactivity of Checkpoint/LoRA to tokens

## What is this?

This is an extension for [stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) which visualizes the reactivity of checkpoint or lora to tokens.

## How to use

1. Select a checkpoint you want to use in usual way. Or, select LoRA in `Reactivity` tab if you want to test LoRA.
2. Click `Run` button in `Reactivity` tab.
3. Wait a few minutes. Progress is displayed in your terminal. If you want to interrupt processing, click `Interrupt` button in `Reactivity` tab.
4. Results will be shown in two tables. The upper one shows tokens with highest reactivity. The lower one shows tokens with lowest reactivity. **`Score` is the highest Frobenius norm of `k^T v` in all cronss-attention layers**.

## Options

- `Num. of output words`: The number of words which will be shown in tables.
- `Batch size`: The number of tokens to be passed to CLIP text encoder at one time.
