# DSPy Lab 🧪

_DSPy for beginners: Programming—not prompting—LMs_

## Overview 📖

This project is a series of labs designed to help you get started with DSPy and RITS. The labs will guide you through the process of initializing language models (LMs), making completion requests, and working with various DSPy functionalities.

## Pre-requisites 📋
- Python 3.10 or higher
- VPN access to the RITS models provider (**TUNNEL REQUIRED**)
- RITS API key (from https://rits.fmaas.res.ibm.com/)

## Installation 💻

1. Clone the repository:
    ```sh
    git clone --branch starter git@github.com:barvhaim/dspy-lab.git
    cd dspy-lab
    ```

2. Create a virtual environment:
    ```sh
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```
   
4. Set up the environment variables:
    ```sh
    export RITS_API_KEY=<your-rits-api>
    export RITS_API_URL=https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com
    ```
   or create a `.env` file in the root directory and add the following:
    ```sh
    RITS_API_KEY=<your-rits-api>
    RITS_API_URL=https://inference-3scale-apicast-production.apps.rits.fmaas.res.ibm.com
    ```

## Usage 🚀

- `starter` branch contains the starter code for the labs. Each lab is implemented in a separate Python file in the `labs` directory.
- Fill in the missing code in each lab to complete the tasks.
- `solution` branch contains the complete code for all the labs.

### Lab 0: Sanity Check ✅

The first lab (`labs/lab_0.py`) initializes a language model and makes a simple completion request to ensure everything is working correctly.

To run the lab:
```sh
python labs/lab_0.py
```

### Lab 1: Signature - Simple 
### Lab 2: Signature - Complex 
### Lab 3: Modules - C-o-T
### Lab 4: Modules - C-o-T with Few-Shot
### Lab 5: Modules - RAG
### Lab 6: Evaluation and Metrics
### Lab 7: Optimization and Tuning (BootstrapFewShot)

