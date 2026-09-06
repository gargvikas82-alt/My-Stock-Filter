name: Stock Hunter v2 Sweep Test (4 configs, one run)

on:
  workflow_dispatch:
    inputs:
      from_date:
        description: 'FROM_DATE - point-in-time date to start scanning from (YYYY-MM-DD)'
        required: true
      to_date:
        description: 'TO_DATE - evaluation date to measure return until (YYYY-MM-DD). Leave blank for today.'
        required: false
        default: ''
      scan_weekdays:
        description: 'Scan days as comma-separated weekday numbers (0=Mon..6=Sun). Default 1,4 = Tue+Fri.'
        required: false
        default: '1,4'

jobs:
  run-sweep:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout Repository
        uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'

      - name: Install Dependencies
        run: |
          python -m pip install --upgrade pip
          pip install pandas yfinance numpy requests

      - name: Run Sweep Test
        env:
          FROM_DATE: ${{ github.event.inputs.from_date }}
          TO_DATE: ${{ github.event.inputs.to_date }}
          SCAN_WEEKDAYS: ${{ github.event.inputs.scan_weekdays }}
        run: |
          python sweep_test.py

      - name: Commit and Push Results
        run: |
          git config --global user.name "github-actions[bot]"
          git config --global user.email "github-actions[bot]@users.noreply.github.com"
          git add sweep_summary.csv sweep_results_*.csv
          git commit -m "Sweep test run [skip ci]" || echo "No changes to commit"
          git push
