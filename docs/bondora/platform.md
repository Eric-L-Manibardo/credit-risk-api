# Bondora: who lends, who funds, who loses

Read this before [note 1](eda1.md). The EDA notes assume the reader
knows how the platform works. This page is that background, tied to
the columns where it shows up in the extract.

## Three parties, not one bank

A bank lends its depositors' money and takes the loss when a borrower
does not pay. Bondora is a **peer-to-peer lending platform** based in
Estonia. It sits between two groups of people:

```
   Borrower  ── applies ──▶  Bondora  ◀── money ──  Investors
      ▲                        │  scores the applicant, sets the rate,
      │                        │  signs the contract, collects payments,
      └── money, then repays ──┘  runs debt collection
```

- **Borrowers** are private people asking for an unsecured consumer
  loan: a typical one is about €2,000 over 60 months. They live in
  Estonia (from 2009), Finland and Spain (from 2013), Slovakia (2014,
  a few hundred loans), or the Netherlands (from 2022).
- **Investors**, many of them private people as well, put up the money.
  Each one usually holds a small slice of many loans.
- **Bondora** decides who gets a loan and at what rate, signs the
  contract, passes the monthly payments to the investors, and runs debt
  collection when a borrower stops paying. It earns fees for that.

The consequence that matters for this repo: **when a loan goes bad, the
investors take the loss, not Bondora.** Recoveries from collection go
back to them (`PrincipalRecovery`, `InterestRecovery`).

## Why the file carries Bondora's own risk estimate

Investors need to judge a loan they did not underwrite. Bondora gives
them its estimate: a letter rating from `AA` (safest) to `HR` (high
risk), a probability of default, and an expected loss (`Rating`,
`ProbabilityOfDefault`, `ExpectedLoss`). In this extract the rating is
filled from 2013 on.

That is Bondora's answer to the question this repo is asking. It is
looked at and kept out of the features ([note 5](eda5.md)), because the
exercise is to build the criterion, not to copy theirs.

## How investors funded a loan, and how that changed

A loan is listed on the platform (`ListedOnUTC`) and investors bid on
it. The extract keeps three kinds of bid, in euros:

| Column | Who placed the bid |
|--------|--------------------|
| `BidsManual` | An investor choosing the loan by hand |
| `BidsPortfolioManager` | Bondora's automatic bidding, following rules the investor set |
| `BidsApi` | An investor's own program, through Bondora's API |

Through 2017 the bids cover the whole loan: the median share of
`Amount` covered by bids is 100% every year. Then it falls: 75% in
2018, 48% in 2019, 3% in 2022, and 0 in 2023-2024. (Shares from
`uv run python scripts/EDA_bondora/bondora_step5_inventory.py`.)

The extract does not say where the rest of the money came from. The
timing matches Go & Grow, a pooled product Bondora opened to invited
investors in April 2018 and to all investors on 5 June 2018
([April 2018 update](https://goandgrow.eu/en/blog/go-grow-major-updates-and-more/),
[release post](https://goandgrow.eu/en/blog/go-and-grow-for-the-dreamers/)).
The investor deposits money and Bondora spreads it across loans, so no
bid is placed on a particular loan. Bondora's 2018 annual report says
the product drove that year's record originations. The company now
trades under the Go & Grow name.

The rate changed hands earlier. Until 2015 the interest rate came out of
the investor auction. In 2015 Bondora replaced it with its own scorecard
and risk-based pricing
([company history](https://help.goandgrow.eu/hc/en-us/articles/14814603348881-What-is-the-history-of-Bondora)).
From then on `Interest` is set by Bondora from its rating, which is why
[note 5](eda5.md) flags it.

So the bid columns describe how a loan was funded in a given year. They
say nothing about the borrower. For the recent book they are close to
empty.

## Words the notes use

- **Origination**: the day the loan is issued (`LoanDate`), and what
  was known then.
- **Vintage**: all loans issued in the same year.
- **Arrears**: payments that are due and unpaid. A loan in arrears shows
  as `Late`.
- **Debt collection**: what Bondora starts when arrears are not cured.
  `DefaultDate` is that day. It is not a write-off: the borrower can
  still pay later ([note 3](eda3.md)).
- **Restructuring**: a change to the payment schedule, usually a longer
  term (`Restructured`, `MaturityDate_Last`).

Next: [what a row is](eda1.md).
