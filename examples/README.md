# RowVoi Examples

Welcome! This directory contains examples to help you learn and use RowVoi effectively. Choose your path based on your needs:

## 🎯 Quick Navigation

| **Your Goal** | **Start Here** | **Description** |
|---------------|----------------|-----------------|
| 🟢 **Learn the basics** | [`getting_started/`](getting_started/) | Start with customer deduplication |
| 🟡 **Solve business problems** | [`business_use_cases/`](business_use_cases/) | Real-world applications |
| 🔴 **Understand algorithms** | [`advanced_algorithms/`](advanced_algorithms/) | Deep technical examples |

## 📁 Directory Structure

```
examples/
├── getting_started/           # 🟢 Perfect for beginners
│   ├── basic_customer_deduplication.py      # Clean Python script
│   └── basic_customer_deduplication.ipynb   # Interactive notebook with explanations
│
├── business_use_cases/        # 🟡 Problem-focused examples
│   └── survey_optimization.py               # Design better questionnaires
│
├── advanced_algorithms/       # 🔴 Technical deep-dives
│   ├── known_data_setcover_demo.py         # Algorithm performance comparisons
│   └── probabilistic_demo.py               # Uncertainty and noise handling
│
└── data/                      # 📊 Shared datasets
    └── customers_sample.csv                 # Real anonymized customer data
```

## 🚀 Getting Started (5 minutes)

### Option 1: Interactive Learning (Recommended)
```bash
# Open the Jupyter notebook
jupyter notebook getting_started/basic_customer_deduplication.ipynb
```

### Option 2: Quick Script
```bash
# Run the Python script
cd getting_started
python basic_customer_deduplication.py
```

## 🎯 Use Case Guide

### 🏢 **Customer/Entity Deduplication**
- **Problem**: Multiple records for the same customer with slight variations
- **Solution**: [`getting_started/basic_customer_deduplication.ipynb`](getting_started/basic_customer_deduplication.ipynb)
- **What you'll learn**: Cost-aware field selection, ROI calculation
- **Business impact**: Save marketing costs, improve customer experience

### 📋 **Survey & Form Optimization** 
- **Problem**: Long surveys with low completion rates
- **Solution**: [`business_use_cases/survey_optimization.py`](business_use_cases/survey_optimization.py)
- **What you'll learn**: Minimize questions while maximizing information
- **Business impact**: Higher response rates, better data quality

### 🔬 **Algorithm Research**
- **Problem**: Need to understand how different algorithms perform
- **Solution**: [`advanced_algorithms/`](advanced_algorithms/) directory
- **What you'll learn**: Performance trade-offs, scalability, uncertainty handling
- **Use case**: Academic research, algorithm selection

## 💡 Key Concepts Explained

### **Minimal Distinguishing Fields**
Find the smallest set of columns that can uniquely identify rows.
```python
key = find_key(df, candidate_rows)  # Returns: ['email', 'phone']
```

### **Interactive Disambiguation**
Guide users through step-by-step questions to resolve ambiguity.
```python
session = DisambiguationSession(df, candidates, policy)
suggestion = session.next_question()  # What to ask next?
```

### **Cost-Aware Selection**
Balance accuracy vs. expense when some fields cost more to verify.
```python
costs = {'email': 1.0, 'address': 5.0, 'income': 10.0}
key = find_key(df, rows, costs=costs)  # Chooses cost-effective fields
```

## 📊 Real Datasets

All examples use realistic, anonymized data:

- **`data/customers_sample.csv`**: Customer database with deliberate duplicates
  - 20 records, ~8 duplicates with variations
  - Fields: name, email, phone, address, company
  - Based on real CRM deduplication scenarios

More datasets will be added based on user feedback.

## 🆘 Getting Help

**If you're new to RowVoi**: Start with the Jupyter notebook in `getting_started/`

**If you have a specific problem**: Check `business_use_cases/` for similar scenarios

**If you need algorithm details**: Explore `advanced_algorithms/`

**If you're stuck**: 
1. Check the [main README](../README.md) for API documentation
2. Look at the docstrings in the code
3. Open an issue on GitHub

## 🎓 Learning Path

1. **Start**: `getting_started/basic_customer_deduplication.ipynb` (15 minutes)
2. **Practice**: Modify the customer dataset and re-run the analysis
3. **Apply**: Try `business_use_cases/survey_optimization.py` (10 minutes)  
4. **Master**: Explore `advanced_algorithms/` based on your interests

## 💬 Feedback

Found these examples helpful? Have suggestions for new use cases? 

- 🐛 **Bug reports**: [GitHub Issues](https://github.com/gojiplus/rowvoi/issues)
- 💡 **Feature requests**: [GitHub Discussions](https://github.com/gojiplus/rowvoi/discussions)
- 📚 **Documentation**: Help us improve these examples!

---

**Happy deduplicating!** 🎉