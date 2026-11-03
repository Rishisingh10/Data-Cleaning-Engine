# Data Cleaning Engine
### An AI-Assisted Data Quality and Cleaning Platform

Data Cleaning Engine is a Python-based application that helps users understand, detect, and resolve data quality issues in CSV and Excel datasets.

The project combines deterministic data processing with LLM-assisted recommendations to create a controlled data-cleaning workflow.

Rather than allowing an AI model to directly modify datasets, the system separates profiling, issue detection, recommendation generation, cleaning execution, and validation.

The goal was to build something more reliable than a collection of data-cleaning scripts: a system that can explain what is wrong with a dataset, recommend what could be done, and let the user decide what actually gets changed.

**The core principle: Detect first. Recommend second. Execute only after approval.**

---

## 1. What does it do?

Data Cleaning Engine takes a CSV or Excel dataset through a structured quality assessment and cleaning process.

| Capability | Description |
|---|---|
| Dataset profiling | Analyses columns, distributions, missing values, and unique values |
| Semantic type inference | Distinguishes identifiers, numerical measurements, categorical data, dates, and other types |
| Missing-value detection | Identifies null and blank values |
| Duplicate detection | Finds exact duplicate records |
| Outlier detection | Uses statistical methods to identify potentially unusual values |
| Formatting detection | Identifies inconsistent categorical representations |
| Mixed-type detection | Identifies columns containing inconsistent data types |
| Invalid-date detection | Identifies values that cannot be interpreted as valid dates |
| AI recommendations | Uses Gemini to suggest appropriate cleaning operations |
| Human approval | Allows individual or complete-plan approval |
| Controlled execution | Executes only predefined cleaning operations |
| Validation | Checks the resulting dataset |
| Audit and history | Maintains cleaning records and execution information |

Supported input formats:
- CSV
- XLSX
- XLSM

Current upload limit: **200 MB**.

---

## 2. The engineering philosophy

One of the most important decisions in this project was separating data detection from data correction.

An unusual value is not necessarily incorrect.

For example:

- A customer ID may be stored as an integer, but calculating its mean makes no sense.
- An unusually high salary may be legitimate.
- A missing value does not always justify deleting an entire row.
- Two category values with different casing may represent the same category, but the original values should not be modified without approval.

This led to a hybrid architecture in which Python performs deterministic operations and the LLM assists with recommendations.

The LLM does not receive permission to execute arbitrary code or directly manipulate the dataset.

---

## 3. System architecture

```text
                  CSV / Excel Upload
                          |
                          v
                    Dataset Loader
                          |
                          v
                    Data Profiler
                          |
                          v
                  Semantic Type Inference
                          |
                          v
                   Quality Detectors
                          |
                          v
                 Structured Quality Report
                          |
                          v
                    Gemini AI Engine
                          |
                          v
                Cleaning Recommendations
                          |
                          v
                    User Approval
                          |
                          v
                 Allowlisted Executor
                          |
                          v
                Validation & Integrity
                          |
                  +-------+-------+
                  |               |
                  v               v
               Audit Log      Cleaned Dataset
                  |
                  v
             Cleaning History
```

### Why a hybrid architecture?

I did not want the entire system to depend on an LLM.

The responsibilities are deliberately separated.

**Python handles:**
- File loading and parsing.
- Statistical profiling.
- Semantic type inference.
- Quality issue detection.
- Cleaning execution.
- Validation and integrity checks.
- Audit records and history.
- Business rule evaluation.

**Gemini handles:**
- Interpreting structured quality findings.
- Generating contextual cleaning recommendations.
- Explaining why a particular cleaning action may be appropriate.

This makes the execution layer deterministic and keeps the AI component replaceable.

---

## 4. Key engineering implementations

### 4.1 Semantic type inference

A column's physical data type does not always represent its actual meaning.

For example:

```text
customer_id
101
102
103
104
```

Although this column contains integers, it represents identifiers rather than numerical measurements.

The profiler maintains physical and semantic type information separately.

Each column profile includes:

- Physical type.
- Semantic type.
- Semantic confidence.
- Inference evidence.
- Total records.
- Non-null count.
- Missing count.
- Unique count.
- Statistical information.
- Sample values.
- Warnings.

This prevents inappropriate statistical operations on identifier columns and improves the quality of downstream detection.

### 4.2 Data quality detection

The detection engine identifies six major categories of data quality issues.

| Issue | Approach |
|---|---|
| Missing values | Null and blank-value inspection |
| Duplicate records | Exact row comparison |
| Potential outliers | Interquartile range (IQR) |
| Mixed data types | Value and semantic consistency checks |
| Inconsistent categorical formatting | Normalised category comparison |
| Invalid dates | Date parsing and validation |

For outlier detection, I implemented the IQR method.

\[
IQR = Q_3-Q_1
\]

\[
Lower = Q_1-1.5(IQR)
\]

\[
Upper = Q_3+1.5(IQR)
\]

Values outside these boundaries are flagged for review.

Importantly, an outlier is not automatically treated as an error.

For categorical formatting, normalised representations are compared to identify differences caused by whitespace or letter casing.

### 4.3 Structured AI recommendations

The Gemini integration receives structured profiling and quality information.

The system uses Pydantic schemas to enforce structured recommendations and reject unexpected fields.

The executor supports a predefined set of operations:

1. Leave unchanged.
2. Drop missing rows.
3. Fill missing values with mean.
4. Fill missing values with median.
5. Fill missing values with mode.
6. Fill missing values with a constant.
7. Normalize whitespace.
8. Standardize case.
9. Remove exact duplicates.
10. Review outliers.

The AI cannot introduce arbitrary executable operations.

Users can approve recommendations individually or approve a complete cleaning plan.

This creates a human-in-the-loop system in which the AI assists with decisions but does not independently control data modifications.

### 4.4 Chunk-based CSV processing

Large CSV files can consume significant memory when loaded entirely into Pandas.

To address this, I implemented a separate chunk-based CSV cleaning executor.

The implementation includes:

- Configurable chunk processing.
- Default chunk size of 50,000 rows.
- Pre-processing passes for global statistics required by cleaning operations.
- SQLite-backed duplicate tracking across chunks.
- Temporary output files.
- Atomic file replacement after successful execution.
- Source-file fingerprint checks.

**Why SQLite for duplicate detection?**

A simple in-memory set can become expensive when processing large datasets.

Also, checking duplicates independently inside each chunk would miss duplicate rows appearing in different chunks.

I used SQLite with a unique constraint on row hashes to maintain duplicate state across chunks.

This allows the executor to check previously encountered rows without retaining the entire duplicate-tracking set in Python memory.

**Important limitation:** Chunk-based processing currently applies to CSV cleaning. Profiling, detection, and AI planning still involve in-memory processing. The entire application is not yet fully out-of-core.

### 4.5 Data integrity and validation

A successful cleaning operation does not automatically mean that the resulting dataset is correct.

During development, I investigated potential data changes caused by CSV parsing and serialization.

One concern was the interpretation of literal values such as `N/A` as missing values.

I adjusted the relevant CSV processing path to use explicit string handling and disable automatic NA interpretation.

I also changed the output-writing approach to use a single open file handle, avoiding repeated encoding and BOM handling issues.

The validation workflow compares input and output records to identify unintended value changes.

**Latest large-dataset validation result:**

| Metric | Result |
|---|---:|
| Input records | 250,000 |
| Output records | 250,000 |
| Value mismatches | 0 |

This result comes from a specific NYC 311 CSV validation run. It does not establish correctness for every possible dataset or CSV encoding.

### 4.6 Business rules and schema drift

The project also includes modules for business rule evaluation and schema drift detection.

Business rules can identify records that violate configured conditions.

The intention is to flag violations for inspection rather than silently modify the original data.

Schema drift detection helps identify structural changes that may affect downstream processing.

### 4.7 Additional modules

Other implemented modules include:

- Privacy inspection.
- Dataset performance assessment.
- Advanced data intelligence.
- Cleaning history.
- Audit management.
- Excel report generation.
- Cleaning validation.
- Business rule construction.

---

## 5. Evaluation and quantitative results

I wanted to evaluate the detection engine at the individual-record level rather than only checking whether an issue category appeared in the final report.

For this, I created synthetic datasets with injected quality problems and a corresponding ground-truth file.

### 5.1 Synthetic benchmark

The row-level evaluation produced the following results:

| Issue category | Precision | Recall | F1-score |
|---|---:|---:|---:|
| Missing values | 100% | 100% | 100% |
| Formatting inconsistencies | 100% | 100% | 100% |
| Potential outliers | 55.56% | 100% | 71.43% |
| Duplicate rows | 100% | 100% | 100% |
| **Macro F1** | — | — | **92.86%** |

These results are from a controlled synthetic benchmark and should not be interpreted as real-world production accuracy.

### 5.2 What the outlier results revealed

The outlier detector achieved:

- Precision: 55.56%
- Recall: 100%
- F1-score: 71.43%

It detected all injected outliers but also flagged eight additional records that were not labelled as outliers in the synthetic ground truth.

This highlighted an important trade-off.

A detector that aggressively identifies unusual values may achieve high recall while generating more false positives.

I retained these results in the evaluation rather than presenting recall alone.

### 5.3 Test suite

The repository includes tests covering:

- Data profiling.
- Cleaning execution.
- Integrity validation.
- Business rules.
- Privacy inspection.
- Performance assessment.
- Audit functionality.
- Schema drift.
- AI failure scenarios.

**75 existing tests passed** during development, as reported during testing.

### 5.4 Large CSV processing validation

I used a public NYC 311 service-request dataset to test processing behaviour on a larger CSV.

| Metric | Result |
|---|---:|
| Records | 250,000 |
| Input format | CSV |
| Chunk size | 50,000 |
| Latest validation mismatches | 0 |

This was a processing and integrity validation exercise, not a labelled detection-accuracy benchmark.

---

## 6. Technology stack

| Component | Technology |
|---|---|
| Programming language | Python |
| Application framework | Streamlit |
| Data processing | Pandas, NumPy |
| AI integration | Google Gemini API |
| Structured validation | Pydantic |
| Duplicate tracking | SQLite |
| Machine learning utilities | Scikit-learn |
| Testing | Pytest |
| Input formats | CSV, XLSX, XLSM |

---

## 7. Project structure

```text
Data-Cleaning-Engine/
│
├── app.py
├── requirements.txt
├── .gitignore
│
├── datadoctor/
│   ├── __init__.py
│   ├── schemas.py
│   ├── loader.py
│   ├── profiler.py
│   ├── detectors.py
│   ├── llm_engine.py
│   ├── cleaning_executor.py
│   ├── chunked_executor.py
│   ├── integrity.py
│   ├── validation.py
│   ├── cleaning_history.py
│   ├── advanced_intelligence.py
│   ├── business_rules.py
│   ├── rule_builder.py
│   ├── report_generator.py
│   ├── privacy.py
│   ├── performance.py
│   ├── audit.py
│   └── schema_drift.py
│
├── evaluation/
│   ├── generate_dataset.py
│   ├── evaluate_detectors.py
│   ├── evaluate_row_level.py
│   ├── benchmark_large_csv.py
│   ├── download_large_nyc.py
│   ├── download_nyc_sample.py
│   ├── validate_benchmark_output.py
│   ├── row_level_results.csv
│   └── test_datasets/
│
└── tests/
    └── ...
```

---

## 8. Running the project locally

### Clone the repository

```bash
git clone https://github.com/Rishisingh10/Data-Cleaning-Engine.git
cd Data-Cleaning-Engine
```

### Create a virtual environment

```bash
python -m venv .venv
```

Activate it:

**Windows**

```powershell
.venv\Scripts\activate
```

**Linux / macOS**

```bash
source .venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Configure the Gemini API key

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_api_key_here
```

Do not upload this file to GitHub.

### Start the application

```bash
streamlit run app.py
```

---

## 9. Current scope and limitations

I deliberately kept the project focused on tabular data instead of expanding it into a general-purpose multimedia processing platform.

Current boundaries:

- CSV and Excel support.
- 200 MB upload limit.
- Chunk-based CSV cleaning.
- In-memory profiling and detection.
- AI-generated recommendations require user review.
- Outlier detection can generate false positives.
- Synthetic evaluation does not establish generalisation to every real-world dataset.
- Audio, image, and video files are not supported.

The priority is to make the existing workflow reliable rather than continuously adding features.

---

## 10. What I learned building this

The most valuable part of this project was understanding where AI should and should not be used.

Some of the engineering lessons:

- Deterministic logic should handle operations that require predictable results.
- LLM recommendations should be structured and constrained.
- Physical data types and semantic types are different.
- Detection and correction should be separate.
- Outliers require context.
- Processing large files requires attention to memory usage and state across chunks.
- Successful execution does not guarantee data integrity.
- Benchmarks should expose false positives as well as successful detections.
- Limitations should be documented instead of hidden.

One of the biggest lessons was that adding complexity does not automatically improve reliability. Some of the most important work involved debugging data transformations and validating what happened to the original values.

---

## 11. Future improvements

Potential areas for further engineering work:

- Out-of-core profiling and detection.
- Better-calibrated anomaly detection.
- Evaluation on independently labelled real-world datasets.
- Broader CSV encoding and Excel edge-case testing.
- More comprehensive memory and throughput benchmarking.

These are possible improvements, not features currently claimed as complete.

---

## 12. About

**Rishi Singh**

B.Tech – Artificial Intelligence and Machine Learning

Interested in AI/ML engineering, data systems, applied machine learning, and building reliable AI-assisted applications.

- GitHub: [Rishisingh10](https://github.com/Rishisingh10)
- Repository: [Data-Cleaning-Engine](https://github.com/Rishisingh10/Data-Cleaning-Engine)

---

*Data Cleaning Engine is an engineering project exploring how deterministic data processing and AI-assisted decision-making can work together while keeping data integrity, traceability, and user control at the centre of the workflow.*
