"""
make_sample_docs.py - generate the demo documents in data/sample_docs/.

The content is written for this project (no copyright issues) and each page has
specific facts, so we can later test that answers cite the right page.

Run:  python scripts/make_sample_docs.py
"""

from pathlib import Path

from fpdf import FPDF

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "sample_docs"

ML_NOTES = {
    "title": "Machine Learning - Study Notes",
    "pages": [
        ("1. What is Machine Learning?",
         "Machine learning (ML) is a branch of artificial intelligence in which computers learn "
         "patterns from data instead of being programmed with explicit rules. A classic definition "
         "by Tom Mitchell (1997) says a program learns from experience E with respect to a task T and "
         "a performance measure P if its performance on T, measured by P, improves with E.\n\n"
         "There are three main types of machine learning. In supervised learning, the model is trained "
         "on labelled examples, where every input comes with the correct output, for example emails "
         "labelled as spam or not spam. In unsupervised learning, the data has no labels and the model "
         "must discover structure on its own, such as grouping customers into segments with clustering. "
         "In reinforcement learning, an agent learns by interacting with an environment and receiving "
         "rewards or penalties, which is how game-playing systems such as AlphaGo were trained.\n\n"
         "A typical ML workflow has six steps: collect data, clean and prepare it, choose a model, train "
         "the model, evaluate it on unseen data, and finally deploy and monitor it. Data preparation "
         "usually takes the most time; practitioners often say it is about 80 percent of the work."),
        ("2. Common Supervised Learning Algorithms",
         "Linear regression predicts a continuous number, such as a house price, by fitting a straight "
         "line (or plane) that minimises the mean squared error between predictions and true values. "
         "It is simple, fast and easy to interpret, which makes it a good first baseline.\n\n"
         "Logistic regression is used for classification, not regression, despite its name. It passes "
         "a linear combination of the inputs through the sigmoid function, which squashes any value into "
         "a probability between 0 and 1. If the probability is above a threshold, usually 0.5, the "
         "example is assigned to the positive class.\n\n"
         "A decision tree splits the data with a series of yes/no questions on the features, choosing "
         "each split to make the resulting groups as pure as possible, measured by Gini impurity or "
         "entropy. Trees are easy to visualise but overfit easily when grown too deep.\n\n"
         "A random forest trains many decision trees on random subsets of the rows and features and "
         "combines their votes. This technique, called bagging, reduces variance and usually gives much "
         "better accuracy than a single tree. The k-nearest neighbours (KNN) algorithm classifies a new "
         "point by the majority label of the k closest training points; it has no training phase but "
         "becomes slow on large datasets."),
        ("3. Overfitting, Underfitting and Validation",
         "Overfitting happens when a model memorises the training data, including its noise, so it scores "
         "very well on training data but poorly on new data. Underfitting is the opposite: the model is "
         "too simple to capture the real pattern and performs badly on both training and test data.\n\n"
         "This is described by the bias-variance trade-off. High bias means the model makes strong, "
         "wrong assumptions and underfits. High variance means the model is too sensitive to small "
         "changes in the training data and overfits. The goal is a model with low total error, which "
         "requires balancing the two.\n\n"
         "Ways to reduce overfitting include collecting more training data, using a simpler model, "
         "regularisation (L1 / Lasso adds the absolute size of the weights to the loss, L2 / Ridge adds "
         "their squares), dropout in neural networks, and early stopping when validation error starts "
         "to rise.\n\n"
         "To measure performance honestly, data is split into training, validation and test sets; a "
         "common split is 70/15/15. In k-fold cross-validation the data is divided into k equal folds "
         "(often k = 5 or 10); the model is trained k times, each time holding out a different fold for "
         "validation, and the k scores are averaged."),
        ("4. Evaluation Metrics for Classification",
         "Accuracy is the share of predictions that are correct. It is misleading on imbalanced data: if "
         "only 1 percent of transactions are fraud, a model that always predicts 'not fraud' is 99 "
         "percent accurate but useless.\n\n"
         "A confusion matrix counts four outcomes: true positives (TP), false positives (FP), true "
         "negatives (TN) and false negatives (FN). Precision = TP / (TP + FP) answers: of everything the "
         "model flagged as positive, how much really was positive? Recall = TP / (TP + FN) answers: of "
         "all real positives, how many did the model find?\n\n"
         "Precision matters most when false alarms are expensive, for example a spam filter that must "
         "not hide important emails. Recall matters most when missing a case is dangerous, for example "
         "cancer screening. The F1 score is the harmonic mean of precision and recall, "
         "F1 = 2 x (precision x recall) / (precision + recall), and is useful when you need a single "
         "number that balances both.\n\n"
         "The ROC curve plots the true positive rate against the false positive rate at every threshold. "
         "The area under it, called AUC, ranges from 0.5 (random guessing) to 1.0 (a perfect classifier)."),
    ],
}

LEAVE_POLICY = {
    "title": "Northwind Tech Pvt. Ltd. - Employee Leave Policy (2026)",
    "pages": [
        ("Section 1: Scope and Annual Leave",
         "This policy applies to all full-time employees of Northwind Tech Pvt. Ltd. who have completed "
         "their probation period of three months. Interns and contract staff follow the separate "
         "Contractor Handbook. The policy is effective from 1 January 2026 and is reviewed every year by "
         "the Human Resources team.\n\n"
         "Every full-time employee is entitled to 18 days of paid annual leave per calendar year. Leave "
         "is credited monthly at the rate of 1.5 days per month. Annual leave must be applied for through "
         "the HR portal at least 7 days in advance, and requests longer than 5 consecutive working days "
         "need approval from both the reporting manager and the department head.\n\n"
         "Unused annual leave can be carried forward to the next year up to a maximum of 10 days. Any "
         "balance above 10 days lapses on 31 December. At the time of resignation, unused annual leave "
         "up to 30 days is paid out (encashed) at the employee's basic salary rate."),
        ("Section 2: Sick Leave and Casual Leave",
         "Employees receive 10 days of paid sick leave per year. Sick leave cannot be carried forward and "
         "cannot be encashed. For sick leave of more than 2 consecutive days, the employee must upload a "
         "medical certificate from a registered doctor to the HR portal within 3 days of returning to "
         "work. Sick leave may also be used to care for an ill spouse, child or parent.\n\n"
         "In addition, employees receive 6 days of casual leave per year for personal matters such as "
         "family events or urgent errands. Casual leave can be taken for at most 2 days at a time and "
         "should be informed to the manager at least one day in advance, except in emergencies.\n\n"
         "If an employee has used all their paid leave, further absence is recorded as leave without pay "
         "(LWP). More than 3 days of unapproved absence in a row may lead to disciplinary action under the "
         "Code of Conduct."),
        ("Section 3: Parental Leave and Work From Home",
         "Female employees are entitled to 26 weeks of paid maternity leave for their first two children, "
         "in line with the Maternity Benefit Act. Up to 8 weeks of this may be taken before the expected "
         "delivery date. Male employees and partners are entitled to 10 working days of paid paternity "
         "leave, which must be used within 3 months of the child's birth or adoption.\n\n"
         "Employees adopting a child below the age of 3 months receive 12 weeks of paid adoption leave.\n\n"
         "Work from home (WFH): after completing probation, employees may work from home up to 2 days per "
         "week, with their manager's approval. WFH days cannot be carried over to the next week. "
         "Employees working from home must be reachable on the company chat tool between 10:00 AM and "
         "6:00 PM.\n\n"
         "For questions about this policy, contact the HR helpdesk at hr-help@northwind.example or "
         "extension 2040."),
    ],
}


def build_pdf(doc: dict, path: Path) -> None:
    """Write one PDF where each (heading, body) pair is a separate page."""
    pdf = FPDF(format="A4")
    pdf.set_margins(20, 20, 20)
    pdf.set_auto_page_break(auto=False)
    for heading, body in doc["pages"]:
        pdf.add_page()
        pdf.set_font("Helvetica", "I", 9)
        pdf.cell(0, 6, doc["title"], new_x="LMARGIN", new_y="NEXT")
        pdf.ln(4)
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 8, heading, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)
        pdf.set_font("Helvetica", "", 11)
        pdf.multi_cell(0, 6, body)
    pdf.output(str(path))


PYTHON_MD = """# Python Data Structures - Quick Reference

## Lists
A list is an ordered, mutable collection written with square brackets, e.g. `[1, 2, 3]`.
Use `append()` to add one item, `extend()` to add many, and `pop()` to remove the last item.
Accessing an item by index is O(1), but searching for a value with `in` is O(n).

## Tuples
A tuple is an ordered, immutable collection written with parentheses, e.g. `(10, 20)`.
Because tuples cannot change, they can be used as dictionary keys and are slightly faster than lists.

## Dictionaries
A dictionary stores key-value pairs, e.g. `{"name": "Asha", "age": 21}`.
Lookups, inserts and deletes are O(1) on average because dictionaries are hash tables.
Since Python 3.7, dictionaries keep the order in which keys were inserted.

## Sets
A set is an unordered collection of unique items, e.g. `{1, 2, 3}`.
Sets are ideal for removing duplicates and for fast membership tests.
They support union (`|`), intersection (`&`) and difference (`-`).

## Choosing the right structure
Use a list for ordered data that changes, a tuple for fixed records, a dictionary for
fast lookup by key, and a set when only uniqueness matters.
"""


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    build_pdf(ML_NOTES, OUT_DIR / "machine_learning_notes.pdf")
    build_pdf(LEAVE_POLICY, OUT_DIR / "company_leave_policy.pdf")
    (OUT_DIR / "python_data_structures.md").write_text(PYTHON_MD, encoding="utf-8")
    print(f"Sample documents written to {OUT_DIR}")
