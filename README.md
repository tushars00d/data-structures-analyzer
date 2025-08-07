# 📊 Data Structures Performance Analyzer

![Streamlit](https://img.shields.io/badge/Built%20With-Streamlit-ff4b4b)
![Python](https://img.shields.io/badge/Language-Python-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

A sleek, interactive, and comprehensive tool built with **Streamlit** to benchmark and analyze the performance of fundamental data structures — **Arrays**, **Linked Lists**, and **HashMaps** — across core operations like **Insert**, **Search**, and **Delete**.  
Gain detailed insights into **execution time**, **memory usage**, and **CPU consumption** with beautiful visualizations and profiler-backed metrics.

---

## 🚀 Features

- 📈 **Visual Performance Analysis**  
  Interactive charts to compare operations across different data structures and input sizes.

- 💻 **Advanced Profiling**  
  Uses `cProfile`, `psutil`, and custom time/memory profilers for accurate analysis.

- 🧠 **Resource Efficiency Insights**  
  Memory, CPU, and time usage visualized to show which data structures are most efficient.

- 📋 **Raw Data Export**  
  View and download raw benchmark data and summary statistics.

- 🎨 **Modern UI**  
  Custom CSS styling for enhanced UX in Streamlit.

---

## 🔍 What’s Benchmarked?

### 🔧 Data Structures:
- **Array (Python List)**
- **Linked List (Custom Implementation)**
- **HashMap (Python Dictionary)**

### ⚙️ Operations:
- `Insert`
- `Search`
- `Delete`

### 📐 Metrics Collected:
- Execution Time (in seconds)
- Memory Usage (MB)
- CPU Usage (%)
- Time Complexity Estimations
- Resource Efficiency Score

---

## 🖥️ Screenshots

<img width="986" height="554" alt="image" src="https://github.com/user-attachments/assets/2fccd2ba-bd63-4e2f-85f0-cf5e07555945" />

<img width="958" height="658" alt="image" src="https://github.com/user-attachments/assets/99bbac6d-9431-4bab-8d3a-d71989aaa538" />

---

## 🧪 Demo

> 💡 Run this locally using the command below:

```
streamlit run app.py
```
---

## 📦 Installation

### 🔗 Requirements

- Python 3.8+
- pip

### 📥 Clone and Install

```
git clone https://github.com/your-username/data-structure-performance-analyzer.git
cd data-structure-performance-analyzer
pip install -r requirements.txt
```

### Required Libraries

```
streamlit
pandas
numpy
matplotlib
seaborn
plotly
memory-profiler
psutil
```

> Or install manually:

```
pip install streamlit pandas numpy matplotlib seaborn plotly memory-profiler psutil
```

---

## ⚙️ How It Works

- Each data structure is benchmarked over user-defined sizes.
- Operations are run in a timed and memory-profiled context.
- Results are collected in a structured format using `@dataclass`.
- Visualizations use **Plotly** and **Streamlit Tabs** for dynamic insights.

---

## 📂 Folder Structure

```
├── app.py                   # Main Streamlit app
├── README.md                # Project documentation
├── requirements.txt         # Python dependencies
└── assets/ (optional)       # Screenshots, visuals
```

---

## 📊 Benchmark Modes

| Mode              | Sizes Tested                       |
|-------------------|-------------------------------------|
| Quick Test        | 100, 500, 1000                     |
| Standard Test     | 1000, 5000, 10000, 20000           |
| Comprehensive     | 1000 → 100000                      |
| Custom            | User-defined                       |

---

## 💡 Use Cases

- Teaching data structure performance
- Interview preparation & visualization
- Exploratory system-level profiling
- Benchmarking custom implementations

---

## 📥 Download Results

- Export full benchmark results as `.csv` for offline analysis.
- Includes execution time, memory delta, CPU usage, complexity info.

---

## 🛡 License

This project is licensed under the **MIT License**.  
Feel free to use, modify, and distribute.

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the repository  
2. Create your feature branch (`git checkout -b feat/feature-name`)  
3. Commit your changes (`git commit -m 'Add new feature'`)  
4. Push to the branch (`git push origin feat/feature-name`)  
5. Open a pull request

---

## 📬 Contact

Created with ❤️ by [Tushar Sood](https://github.com/tushars00d)  
For any queries or feedback, feel free to connect via [LinkedIn](https://www.linkedin.com/in/tushars00d)

---

## 🌟 Show Your Support

If you liked this project, feel free to ⭐ the repository and share it with others!
