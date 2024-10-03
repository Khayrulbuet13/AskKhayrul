#!/bin/bash

pip install requests
# Run any setup steps or pre-processing tasks here
echo "Running ETL to move chatbot data from csvs to Neo4j..."

# Run the ETL script
python education_csv_write.py