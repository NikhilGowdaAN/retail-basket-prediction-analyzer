# Retail Basket Prediction Analyzer

[![Built with askThili](https://ask.thili.ai/api/v1/public/projects/463vb3qpyx/badge.svg)](https://ask.thili.ai/b/463vb3qpyx)

This project implements a Retail Basket Prediction Analyzer. It allows retail analysts to compare the next-item prediction accuracy between an n-gram model and a small language model. Both models are trained on simulated shopping basket data, providing insights into their respective strengths for predicting customer purchasing behavior.

## What you are building

**Goal.** You will build a tool that trains and evaluates both an n-gram model and a small language model on simulated shopping basket data to compare their next-item prediction accuracy.

**Done when.** You will have trained both a next-item language model and an N-gram model on the same simulated retail basket data, evaluated their prediction performance using metrics like accuracy or rank, and presented a comparison of their results.

## Milestones

Tick these off as you go, and commit after each one.

- [x] **1. Generate a synthetic dataset of shopping baskets with planted itemsets.**  
      Hint: Use the data generation technique from "Teach a Language Model What Sells Together".
- [ ] **2. Implement the n-gram model to predict next items in a basket.**  
      Hint: Apply the n-gram model building technique from "Build a Next-Word Predictor from Scratch".
- [ ] **3. Pretrain a small language model (like in the drill) on the basket sequences.**  
      Hint: Adapt the language model training process from "Teach a Language Model What Sells Together".
- [ ] **4. Evaluate both models using next-item prediction metrics (e.g., accuracy, rank).**  
      Hint: Calculate standard next-item prediction metrics like accuracy or rank.
- [ ] **5. Present a comparison of the models' performance.**  
      Hint: Summarize and compare the evaluation results from both models.

## Run it

You need Python 3.12 and git.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then choose a model; see the comments inside
python -m app.cli hello
pytest
```

Your work goes in `app/core.py` (one stub per milestone) and `tests/test_core.py`. The first tests pass already; they
check the plumbing.

## Settings and secrets

Model settings live in `.env`, which is ignored by git. **Never commit `.env` or paste a key into code.** If a key ever
reaches GitHub, treat it as leaked and make a new one.

## Stretch ideas

- Explore how varying the 'n' in the n-gram model impacts its performance.
- Analyze specific types of shopping baskets (e.g., those with strong itemset patterns vs. weaker ones) to see if one model performs better.
- Visualize the prediction errors or confidence scores for both models to gain deeper insights.
