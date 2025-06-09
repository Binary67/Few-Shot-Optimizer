# Few-Shot Prompt Optimizer

A Python program that automatically optimizes prompts by intelligently selecting the best few-shot examples to improve model accuracy on your specific tasks.

## What Does This Do?

This tool helps you build better prompts by:
- Testing different combinations of example demonstrations (few-shot examples)
- Measuring which examples actually improve your model's performance
- Only adding examples that increase accuracy on your validation data
- Building an optimized prompt automatically

Think of it like A/B testing for your prompts - it tries different combinations and keeps only what works.

## How It Works

1. **Start with a base prompt** - Your initial prompt without examples
2. **Load your training data** - Examples with inputs and expected outputs
3. **Test systematically** - Try adding different few-shot examples one by one
4. **Measure performance** - Run validation tests after each addition
5. **Keep improvements only** - Only add examples that actually boost accuracy
6. **Build final prompt** - Return your optimized prompt with the best examples
