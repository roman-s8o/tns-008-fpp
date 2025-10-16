# Sentiment Analysis Validation Report

**Generated**: 2025-10-16 15:25:03

**Milestone**: 13 - Feature Extraction (Sentiment)

---

## Summary Statistics

**Total Articles Analyzed**: 30

**Method Agreement Rate**: 46.7%

### FinBERT-Sentiment Distribution

| Label | Count | Percentage |
|-------|-------|------------|
| Neutral | 15 | 50.0% |
| Negative | 9 | 30.0% |
| Positive | 6 | 20.0% |

### SSL-Embeddings Distribution

| Label | Count | Percentage |
|-------|-------|------------|
| Negative | 17 | 56.7% |
| Positive | 13 | 43.3% |

---

## Sample Predictions

Below are sample articles with sentiment predictions from both methods.

---

### Example 1

**Headline**: Mortgage and refinance interest rates today, October 12, 2025: Best week of the year to buy a house

**Content**: 

**Predictions**:

| Method | Label | Score | Confidence |
|--------|-------|-------|------------|
| FinBERT-Sentiment | **Neutral** | 0.144 | 0.831 |
| SSL-Embeddings | **Positive** | 0.038 | 0.014 |

⚠️ **Methods Disagree**

**Average Sentiment Score**: 0.091

---

### Example 2

**Headline**: I’m having brain surgery. Will my heirs or the creditors get my money if I die?

**Content**: “I’m 47 and have a small amount of investments totaling about $45,000, and I also have the same amount in debt.”

**Predictions**:

| Method | Label | Score | Confidence |
|--------|-------|-------|------------|
| FinBERT-Sentiment | **Neutral** | -0.096 | 0.855 |
| SSL-Embeddings | **Negative** | -0.001 | 0.000 |

⚠️ **Methods Disagree**

**Average Sentiment Score**: -0.049

---

### Example 3

**Headline**: Why Income Investors See McCormick & Company (MKC) as a Reliable Choice Among Food Dividend Stocks

**Content**: 

**Predictions**:

| Method | Label | Score | Confidence |
|--------|-------|-------|------------|
| FinBERT-Sentiment | **Positive** | 0.589 | 0.599 |
| SSL-Embeddings | **Positive** | 0.072 | 0.028 |

✅ **Methods Agree**

**Average Sentiment Score**: 0.331

---

### Example 4

**Headline**: California woman, 69, seeks Dave Ramsey’s advice after her husband lost their entire $1M nest egg sports gambling

**Content**: 

**Predictions**:

| Method | Label | Score | Confidence |
|--------|-------|-------|------------|
| FinBERT-Sentiment | **Neutral** | -0.397 | 0.511 |
| SSL-Embeddings | **Negative** | -0.079 | 0.031 |

⚠️ **Methods Disagree**

**Average Sentiment Score**: -0.238

---

### Example 5

**Headline**: Activist Irenic takes a stake in Atkore, urges company to consider a sale

**Content**: Irenic's stake comes at a critical inflection point for Atkore. It needs a CEO, has operational and capital challenges, and a poor market perception.

**Predictions**:

| Method | Label | Score | Confidence |
|--------|-------|-------|------------|
| FinBERT-Sentiment | **Negative** | -0.938 | 0.948 |
| SSL-Embeddings | **Negative** | -0.085 | 0.033 |

✅ **Methods Agree**

**Average Sentiment Score**: -0.512

---

### Example 6

**Headline**: China 'not afraid' of trade war with U.S.

**Content**: 

**Predictions**:

| Method | Label | Score | Confidence |
|--------|-------|-------|------------|
| FinBERT-Sentiment | **Neutral** | 0.143 | 0.716 |
| SSL-Embeddings | **Positive** | 0.011 | 0.004 |

⚠️ **Methods Disagree**

**Average Sentiment Score**: 0.077

---

### Example 7

**Headline**: I've studied over 200 kids—parents who raise mentally strong children never do 7 things

**Content**: It's our job to give kids the tools they need to be mentally strong in life. Child psychologist Reem Raouda has studied more than 200 child-parent relationships. Based on her research, she's identified the seven things to avoid if you want your kids to be resilient and confident.

**Predictions**:

| Method | Label | Score | Confidence |
|--------|-------|-------|------------|
| FinBERT-Sentiment | **Neutral** | -0.027 | 0.883 |
| SSL-Embeddings | **Negative** | -0.006 | 0.002 |

⚠️ **Methods Disagree**

**Average Sentiment Score**: -0.016

---

### Example 8

**Headline**: ‘I can’t stop thinking about money’: I’m struggling with guilt over my $135K in financial mistakes. How do I move on?

**Content**: “We aren’t in debt outside of our mortgage, but I feel sick when I think about the money.”

**Predictions**:

| Method | Label | Score | Confidence |
|--------|-------|-------|------------|
| FinBERT-Sentiment | **Negative** | -0.681 | 0.719 |
| SSL-Embeddings | **Negative** | -0.071 | 0.028 |

✅ **Methods Agree**

**Average Sentiment Score**: -0.376

---

### Example 9

**Headline**: Wall Street Has ‘White Knuckle Moment’ After Tariff Threat Sends Markets Reeling

**Content**: 

**Predictions**:

| Method | Label | Score | Confidence |
|--------|-------|-------|------------|
| FinBERT-Sentiment | **Negative** | -0.318 | 0.562 |
| SSL-Embeddings | **Negative** | -0.070 | 0.028 |

✅ **Methods Agree**

**Average Sentiment Score**: -0.194

---

### Example 10

**Headline**: Trump says administration has 'identified funds' to pay troops during shutdown

**Content**: President Donald Trump on Saturday said his administration has "identified funds" to pay troops next week despite the federal government shutdown.

**Predictions**:

| Method | Label | Score | Confidence |
|--------|-------|-------|------------|
| FinBERT-Sentiment | **Positive** | 0.781 | 0.813 |
| SSL-Embeddings | **Negative** | -0.006 | 0.002 |

⚠️ **Methods Disagree**

**Average Sentiment Score**: 0.387

---

### Example 11

**Headline**: The Best Growth Stock to Invest $1,000 in Right Now

**Content**: 

**Predictions**:

| Method | Label | Score | Confidence |
|--------|-------|-------|------------|
| FinBERT-Sentiment | **Neutral** | 0.123 | 0.860 |
| SSL-Embeddings | **Positive** | 0.059 | 0.023 |

⚠️ **Methods Disagree**

**Average Sentiment Score**: 0.091

---

### Example 12

**Headline**: Spirit Airlines wins approval for $475 million lifeline in bankruptcy court

**Content**: Spirit Airlines will get a $475 million lifeline from its bondholders and another $150 million from an aircraft lessor.

**Predictions**:

| Method | Label | Score | Confidence |
|--------|-------|-------|------------|
| FinBERT-Sentiment | **Positive** | 0.598 | 0.616 |
| SSL-Embeddings | **Positive** | 0.057 | 0.022 |

✅ **Methods Agree**

**Average Sentiment Score**: 0.328

---

### Example 13

**Headline**: Trump threatens 'massive' tariff hike on China over rare earths dispute

**Content**: Stock markets dropped on Trump's bellicose Truth Social post that said China is "becoming very hostile" in seeking tough export controls on rare earths.

**Predictions**:

| Method | Label | Score | Confidence |
|--------|-------|-------|------------|
| FinBERT-Sentiment | **Negative** | -0.950 | 0.964 |
| SSL-Embeddings | **Negative** | -0.132 | 0.050 |

✅ **Methods Agree**

**Average Sentiment Score**: -0.541

---

### Example 14

**Headline**: Netflix Stock Still Looks 15% Too Cheap, Especially If It Keeps Producing 20% FCF Margins

**Content**: 

**Predictions**:

| Method | Label | Score | Confidence |
|--------|-------|-------|------------|
| FinBERT-Sentiment | **Negative** | -0.670 | 0.738 |
| SSL-Embeddings | **Negative** | -0.026 | 0.010 |

✅ **Methods Agree**

**Average Sentiment Score**: -0.348

---

### Example 15

**Headline**: Here are the 4 big things we're watching in the stock market in the week ahead

**Content**: President Trump ratcheted up China tensions on the eve of third-quarter earnings season.

**Predictions**:

| Method | Label | Score | Confidence |
|--------|-------|-------|------------|
| FinBERT-Sentiment | **Negative** | -0.362 | 0.455 |
| SSL-Embeddings | **Negative** | -0.006 | 0.002 |

✅ **Methods Agree**

**Average Sentiment Score**: -0.184

---

### Example 16

**Headline**: Trump talks tough with China but holds out hope of truce in trade war

**Content**: US hardliners want a tough stance but the US president is taking a more nuanced approach to Beijing

**Predictions**:

| Method | Label | Score | Confidence |
|--------|-------|-------|------------|
| FinBERT-Sentiment | **Neutral** | 0.170 | 0.422 |
| SSL-Embeddings | **Negative** | -0.016 | 0.006 |

⚠️ **Methods Disagree**

**Average Sentiment Score**: 0.077

---

### Example 17

**Headline**: Govini, a defense tech startup taking on Palantir, hits $100 million in annual recurring revenue

**Content**: Defense tech startup Govini said it has surpassed $100 million in annual recurring revenue and secured a $150 million investment from Bain Capital.

**Predictions**:

| Method | Label | Score | Confidence |
|--------|-------|-------|------------|
| FinBERT-Sentiment | **Positive** | 0.922 | 0.939 |
| SSL-Embeddings | **Positive** | 0.025 | 0.010 |

✅ **Methods Agree**

**Average Sentiment Score**: 0.474

---

### Example 18

**Headline**: Tesla (TSLA) Target Boosted to $509 by TD Cowen After Strong Deliveries and AI Momentum

**Content**: 

**Predictions**:

| Method | Label | Score | Confidence |
|--------|-------|-------|------------|
| FinBERT-Sentiment | **Positive** | 0.923 | 0.948 |
| SSL-Embeddings | **Positive** | 0.011 | 0.004 |

✅ **Methods Agree**

**Average Sentiment Score**: 0.467

---

### Example 19

**Headline**: I’m 58, divorced and will retire at 60 with $5,300 a month. Is now a good time to buy a house?

**Content**: “Due to my two divorces, I had to leave the homes I bought and give up all the equity.”

**Predictions**:

| Method | Label | Score | Confidence |
|--------|-------|-------|------------|
| FinBERT-Sentiment | **Neutral** | -0.053 | 0.873 |
| SSL-Embeddings | **Negative** | -0.022 | 0.009 |

⚠️ **Methods Disagree**

**Average Sentiment Score**: -0.038

---

### Example 20

**Headline**: HELOC rates today, October 12, 2025: Rates fall 19 basis points in 3 months

**Content**: 

**Predictions**:

| Method | Label | Score | Confidence |
|--------|-------|-------|------------|
| FinBERT-Sentiment | **Negative** | -0.963 | 0.972 |
| SSL-Embeddings | **Negative** | -0.116 | 0.044 |

✅ **Methods Agree**

**Average Sentiment Score**: -0.540

---

### Example 21

**Headline**: Kevin Warsh Says Jerome Powell Has Failed. Inside the Mind of the Man Who May Lead the Trump Fed.

**Content**: 

**Predictions**:

| Method | Label | Score | Confidence |
|--------|-------|-------|------------|
| FinBERT-Sentiment | **Neutral** | -0.000 | 0.874 |
| SSL-Embeddings | **Positive** | 0.016 | 0.006 |

⚠️ **Methods Disagree**

**Average Sentiment Score**: 0.008

---

### Example 22

**Headline**: Starmer and Macron to join Trump at Gaza ‘peace summit’ in Egypt

**Content**: Monday meeting comes as Hamas is due to free Israeli hostages in exchange for Palestinian prisoners

**Predictions**:

| Method | Label | Score | Confidence |
|--------|-------|-------|------------|
| FinBERT-Sentiment | **Neutral** | 0.078 | 0.841 |
| SSL-Embeddings | **Negative** | -0.024 | 0.009 |

⚠️ **Methods Disagree**

**Average Sentiment Score**: 0.027

---

### Example 23

**Headline**: ‘I’d hate to turn her over to the state’: My mother, who has dementia, refuses to be put in a facility. What can we do?

**Content**: “She has never been willing to sign a power of attorney.”

**Predictions**:

| Method | Label | Score | Confidence |
|--------|-------|-------|------------|
| FinBERT-Sentiment | **Neutral** | -0.403 | 0.519 |
| SSL-Embeddings | **Negative** | -0.068 | 0.027 |

⚠️ **Methods Disagree**

**Average Sentiment Score**: -0.236

---

### Example 24

**Headline**: Market sell-off: Trump post lops off $2 trillion from stocks in a single day

**Content**: The unraveling shows the sway the president's one-man trade policy still has over the fate of the global economy.

**Predictions**:

| Method | Label | Score | Confidence |
|--------|-------|-------|------------|
| FinBERT-Sentiment | **Negative** | -0.898 | 0.920 |
| SSL-Embeddings | **Negative** | -0.033 | 0.013 |

✅ **Methods Agree**

**Average Sentiment Score**: -0.465

---

### Example 25

**Headline**: A CD Ladder Is the Right Step for These Young Workers. Here’s Why.

**Content**: 

**Predictions**:

| Method | Label | Score | Confidence |
|--------|-------|-------|------------|
| FinBERT-Sentiment | **Neutral** | 0.054 | 0.900 |
| SSL-Embeddings | **Positive** | 0.033 | 0.013 |

⚠️ **Methods Disagree**

**Average Sentiment Score**: 0.044

---

### Example 26

**Headline**: China's lesson for the US: it takes more than chips to win the AI race

**Content**: 

**Predictions**:

| Method | Label | Score | Confidence |
|--------|-------|-------|------------|
| FinBERT-Sentiment | **Neutral** | 0.124 | 0.850 |
| SSL-Embeddings | **Positive** | 0.056 | 0.022 |

⚠️ **Methods Disagree**

**Average Sentiment Score**: 0.090

---

### Example 27

**Headline**: Dow Jones Futures Loom As Trump Says 'All Will Be Fine' After Imposing 100% China Tariff

**Content**: 

**Predictions**:

| Method | Label | Score | Confidence |
|--------|-------|-------|------------|
| FinBERT-Sentiment | **Negative** | -0.329 | 0.493 |
| SSL-Embeddings | **Negative** | -0.026 | 0.010 |

✅ **Methods Agree**

**Average Sentiment Score**: -0.177

---

### Example 28

**Headline**: AMD Amps Up Chip War

**Content**: 

**Predictions**:

| Method | Label | Score | Confidence |
|--------|-------|-------|------------|
| FinBERT-Sentiment | **Neutral** | 0.102 | 0.670 |
| SSL-Embeddings | **Positive** | 0.020 | 0.008 |

⚠️ **Methods Disagree**

**Average Sentiment Score**: 0.061

---

### Example 29

**Headline**: BMO Maintains Outperform on Amazon (AMZN), Calls It a Top Pick

**Content**: 

**Predictions**:

| Method | Label | Score | Confidence |
|--------|-------|-------|------------|
| FinBERT-Sentiment | **Positive** | 0.544 | 0.731 |
| SSL-Embeddings | **Positive** | 0.053 | 0.021 |

✅ **Methods Agree**

**Average Sentiment Score**: 0.298

---

### Example 30

**Headline**: Morgan Stanley is opening cryptocurrency investments to all clients. Here’s what percentage of your portfolio should be in crypto.

**Content**: Morgan Stanley will rely on its automated monitoring processes to make sure clients are not overly exposed to crypto.

**Predictions**:

| Method | Label | Score | Confidence |
|--------|-------|-------|------------|
| FinBERT-Sentiment | **Neutral** | 0.066 | 0.909 |
| SSL-Embeddings | **Positive** | 0.061 | 0.024 |

⚠️ **Methods Disagree**

**Average Sentiment Score**: 0.063

---

## Method Comparison

**Agreement Rate**: 46.7%

**Average Sentiment Scores**:

| Method | Mean | Std Dev |
|--------|------|--------|
| FinBERT-Sentiment | -0.057 | 0.539 |
| SSL-Embeddings | -0.009 | 0.054 |
| Average | -0.033 | 0.291 |

**Confidence Scores**:

| Method | Mean | Std Dev |
|--------|------|--------|
| FinBERT-Sentiment | 0.764 | 0.170 |
| SSL-Embeddings | 0.017 | 0.013 |

---

## Recommendations

❌ **Low Agreement**: Methods only agree 46.7% of the time.

**Recommendation**: Review sentiment extraction logic. Consider using only FinBERT-Sentiment.

---

## Next Steps

1. Review sample predictions for accuracy
2. Validate sentiment scores against known market events
3. Proceed to Milestone 14: Feature Extraction (NER, Topics)
4. Use sentiment features in downstream prediction tasks

