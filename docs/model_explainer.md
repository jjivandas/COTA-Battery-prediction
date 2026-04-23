# Model Explainer In Simple English

## The big idea

We are not using multiple models because more models always means better science.

We are using multiple models because each one answers a different question:
- Is the pattern mostly linear?
- Do we need nonlinear behavior?
- Are we learning something real or just memorizing?
- Does performance hold up across buses and over time?

Simple way to explain it:
- "We start simple and only add complexity when the simpler model leaves something unexplained."

## The two targets

### 1. Cumulative target

This is `avg_qloss`.

What it means:
- total battery degradation up to that week

Why it matters:
- it tells us how much degradation has built up overall

Limitation:
- it is easier to predict because it mostly grows over time

Simple way to explain it:
- "This tells us how much damage has accumulated so far."

### 2. Delta target

This is `delta_qloss`.

What it means:
- how much new degradation happened this week

Why it matters:
- this is better for understanding what weekly operating conditions drive degradation

Simple way to explain it:
- "This tells us how much extra damage happened this week, not just the total so far."

## Why we start with linear regression

### What linear regression is

Linear regression is the simplest model in the set.

It assumes:
- the target can be explained as a weighted combination of the input features

Simple way to explain it:
- "It draws the best straight-line style relationship through the data."

### Why we use it first

We use it first because:
- it is easy to interpret
- it gives a clean baseline
- if it already works well, we may not need a more complex model

What it tells us if it does well:
- the main relationships may be mostly linear
- the features already carry useful signal

What it tells us if it struggles:
- the relationship may be nonlinear
- the features may interact in more complicated ways

Simple way to explain it:
- "If the simple model already works, that is a strong result because it means the pattern is clean and understandable."

## Why we try random forest after linear regression

### What random forest is

Random forest is a collection of many decision trees whose predictions are averaged together.

It can capture:
- nonlinear effects
- feature interactions
- threshold behavior

Simple way to explain it:
- "Instead of fitting one straight line, it learns many if-then rules and averages them."

### Why it comes after linear regression

Random forest is the natural next step because it tells us whether there is useful complexity beyond the linear baseline.

What it tells us if it beats linear regression clearly:
- the data contains nonlinear structure
- some effects depend on combinations of variables, not just one variable at a time

What it tells us if it does not help much:
- the simpler linear explanation may already be enough

Simple way to explain it:
- "We use random forest to check whether the battery story is more complicated than a straight-line model can capture."

## Why we compare both models on both targets

Running one model on one target is not enough.

The full set gives a better story:
- linear + cumulative: simple baseline for total degradation
- linear + delta: simple baseline for week-to-week degradation
- random forest + cumulative: checks for nonlinear total-degradation patterns
- random forest + delta: checks for nonlinear weekly drivers

Simple way to explain it:
- "Together, these runs tell us whether the signal is simple or complex, and whether that is true for both total degradation and weekly degradation."

## What each evaluation split means

### GroupKFold by bus

Question it answers:
- can the model generalize to buses it has never seen before?

Simple way to explain it:
- "If a new bus shows up, does the model still work?"

### Leave-buses-out holdout

Question it answers:
- does the model still look good on a final untouched set of buses?

Simple way to explain it:
- "This is our more honest final exam."

### Temporal split

Question it answers:
- can the model predict future weeks from earlier weeks?

Simple way to explain it:
- "Can the model forecast what happens next, not just explain what it has already seen?"

## Why all the evaluations matter together

A model can look strong in one test and weak in another.

Examples:
- good cross-bus score but bad temporal score means it generalizes across buses better than it forecasts the future
- good temporal score but weak unseen-bus score means it may fit known buses better than new ones

Simple way to explain it:
- "One score never tells the whole story. We need all three views."

## What comes after these models

If the simple and nonlinear baselines are not enough, later models help answer new questions.

### Ridge and Lasso

Why use them:
- stabilize linear modeling
- reduce overfitting
- help show which features matter most

Simple way to explain it:
- "These are cleaner, more controlled versions of linear regression."

### Gradient Boosting

Why use it:
- often performs better than random forest on tabular data
- captures strong nonlinear structure more efficiently

Simple way to explain it:
- "This is a stronger nonlinear model if random forest suggests complexity is important."

### SVR, Bayesian Ridge, Gaussian Process

Why use them:
- test different mathematical assumptions
- add uncertainty-aware modeling in some cases

Simple way to explain it:
- "These help if we want different geometry, better regularization, or uncertainty estimates."

## Short slide version

- We start with linear regression because it is simple and interpretable.
- We add random forest to test whether nonlinear behavior matters.
- We test both total degradation and weekly degradation.
- We score each model three ways: unseen buses, untouched holdout buses, and future weeks.
- This gives us a balanced picture of accuracy, robustness, and forecasting value.
