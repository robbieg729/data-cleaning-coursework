import pandas as pd

# Load data frames from different years, not including the text responses from 2019
mcq_2019 = pd.read_csv("Kaggle/kaggle-survey/kaggle-survey-2019/multiple_choice_responses.csv")
mcq_2020 = pd.read_csv("Kaggle/kaggle-survey/kaggle-survey-2020/kaggle_survey_2020_responses.csv")
mcq_2021 = pd.read_csv("Kaggle/kaggle-survey/kaggle-survey-2021/kaggle_survey_2021_responses.csv")

mcq_2019_cols = mcq_2019.iloc[0] # Get first row, which is the actual questions
mcq_2019.drop(axis=0, index=0, inplace=True) # Drop the first row, which contains the actual questions
mcq_2019.columns = mcq_2019_cols # Set columns equal to the actual questions

mcq_2020_cols = mcq_2020.iloc[0] # Get first row, which is the actual questions
mcq_2020.drop(axis=0, index=0, inplace=True) # Drop the first row, which contains the actual questions
mcq_2020.columns = mcq_2020_cols # Set columns equal to the actual questions

mcq_2021_cols = mcq_2021.iloc[0] # Get first row, which is the actual questions
mcq_2021.drop(axis=0, index=0, inplace=True) # Drop the first row, which contains the actual questions
mcq_2021.columns = mcq_2021_cols # Set columns equal to the actual questions

def combine_multiple_option_columns(columns, df):
    col_1 = columns[0] # column of first option
    col_2 = columns[1] # column of second option
    df = df[columns].fillna("") # fill all NaN values with empty string in the relevant columns
    combined = pd.Series(df[col_1] + df[col_2]) # string concatenate the two columns so that any 
    # rows which were both originally NaN will be the empty string, and any rows that had at least
    # one option selected will have a non-empty value. https://www.statology.org/pandas-combine-two-columns/
    return list(combined) # return the list of the concatenated columns

# Merge options for the given columns into one column 
mcq_2019["Which of the following cloud computing platforms do you use on a regular basis? (Select all that apply) - Selected Choice -  IBM Cloud "] = combine_multiple_option_columns(
        ["Which of the following cloud computing platforms do you use on a regular basis? (Select all that apply) - Selected Choice -  IBM Cloud ",
        "Which of the following cloud computing platforms do you use on a regular basis? (Select all that apply) - Selected Choice -  Red Hat Cloud "],
        mcq_2019)

# Rename some of the columns to match with similar ones (i.e. ones asking the same question) in the other surveys
mcq_2019.rename(columns={"Approximately how much money have you spent on machine learning and/or cloud computing products at your work in the past 5 years?": "Approximately how much money have you (or your team) spent on machine learning and/or cloud computing services at home (or at work) in the past 5 years (approximate $USD)?",
                        "Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - Reddit (r/machinelearning, r/datascience, etc)": "Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - Reddit (r/machinelearning, etc)",
                        "Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - Kaggle (forums, blog, social media, etc)": "Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - Kaggle (notebooks, forums, etc)",
                        "Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - Course Forums (forums.fast.ai, etc)": "Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - Course Forums (forums.fast.ai, Coursera forums, etc)",
                        "Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - YouTube (Cloud AI Adventures, Siraj Raval, etc)": "Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - YouTube (Kaggle YouTube, Cloud AI Adventures, etc)",
                        "Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - Podcasts (Chai Time Data Science, Linear Digressions, etc)": "Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - Podcasts (Chai Time Data Science, Oâ€™Reilly Data Show, etc)",
                        "Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - Blogs (Towards Data Science, Medium, Analytics Vidhya, KDnuggets etc)": "Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - Blogs (Towards Data Science, Analytics Vidhya, etc)",
                        "Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - Journal Publications (traditional publications, preprint journals, etc)": "Who/what are your favorite media sources that report on data science topics? (Select all that apply) - Selected Choice - Journal Publications (peer-reviewed journals, conference proceedings, etc)",
                        "How long have you been writing code to analyze data (at work or at school)?" : "For how many years have you been writing code and/or programming?",
                        "Which of the following hosted notebook products do you use on a regular basis?  (Select all that apply) - Selected Choice -  Kaggle Notebooks (Kernels) ": "Which of the following hosted notebook products do you use on a regular basis?  (Select all that apply) - Selected Choice -  Kaggle Notebooks",
                        "Which of the following hosted notebook products do you use on a regular basis?  (Select all that apply) - Selected Choice -  Google Colab ": "Which of the following hosted notebook products do you use on a regular basis?  (Select all that apply) - Selected Choice - Colab Notebooks",
                        "Which of the following hosted notebook products do you use on a regular basis?  (Select all that apply) - Selected Choice -  Microsoft Azure Notebooks ": "Which of the following hosted notebook products do you use on a regular basis?  (Select all that apply) - Selected Choice - Azure Notebooks",
                        "Which of the following hosted notebook products do you use on a regular basis?  (Select all that apply) - Selected Choice - Code Ocean ": "Which of the following hosted notebook products do you use on a regular basis?  (Select all that apply) - Selected Choice -  Code Ocean ",
                        "Which types of specialized hardware do you use on a regular basis?  (Select all that apply) - Selected Choice - None / I do not know": "Which types of specialized hardware do you use on a regular basis?  (Select all that apply) - Selected Choice - None",
                        "Have you ever used a TPU (tensor processing unit)?": "Approximately how many times have you used a TPU (tensor processing unit)?",
                        "Which of the following ML algorithms do you use on a regular basis? (Select all that apply): - Selected Choice - Transformer Networks (BERT, gpt-2, etc)": "Which of the following ML algorithms do you use on a regular basis? (Select all that apply): - Selected Choice - Transformer Networks (BERT, gpt-3, etc)",
                        "Which categories of ML tools do you use on a regular basis?  (Select all that apply) - Selected Choice - Automated data augmentation (e.g. imgaug, albumentations)": "Do you use any automated machine learning tools (or partial AutoML tools) on a regular basis?  (Select all that apply) - Selected Choice - Automated data augmentation (e.g. imgaug, albumentations)",
                        "Which categories of ML tools do you use on a regular basis?  (Select all that apply) - Selected Choice - Automated feature engineering/selection (e.g. tpot, boruta_py)": "Do you use any automated machine learning tools (or partial AutoML tools) on a regular basis?  (Select all that apply) - Selected Choice - Automated feature engineering/selection (e.g. tpot, boruta_py)",
                        "Which categories of ML tools do you use on a regular basis?  (Select all that apply) - Selected Choice - Automated model selection (e.g. auto-sklearn, xcessiv)": "Do you use any automated machine learning tools (or partial AutoML tools) on a regular basis?  (Select all that apply) - Selected Choice - Automated model selection (e.g. auto-sklearn, xcessiv)",
                        "Which categories of ML tools do you use on a regular basis?  (Select all that apply) - Selected Choice - Automated model architecture searches (e.g. darts, enas)": "Do you use any automated machine learning tools (or partial AutoML tools) on a regular basis?  (Select all that apply) - Selected Choice - Automated model architecture searches (e.g. darts, enas)",
                        "Which categories of ML tools do you use on a regular basis?  (Select all that apply) - Selected Choice - Automated hyperparameter tuning (e.g. hyperopt, ray.tune)": "Do you use any automated machine learning tools (or partial AutoML tools) on a regular basis?  (Select all that apply) - Selected Choice - Automated hyperparameter tuning (e.g. hyperopt, ray.tune, Vizier)",
                        "Which categories of ML tools do you use on a regular basis?  (Select all that apply) - Selected Choice - Automation of full ML pipelines (e.g. Google AutoML, H20 Driverless AI)": "Do you use any automated machine learning tools (or partial AutoML tools) on a regular basis?  (Select all that apply) - Selected Choice - Automation of full ML pipelines (e.g. Google AutoML, H2O Driverless AI)",
                        "Which categories of ML tools do you use on a regular basis?  (Select all that apply) - Selected Choice - None": "Do you use any automated machine learning tools (or partial AutoML tools) on a regular basis?  (Select all that apply) - Selected Choice - No / None",
                        "Which categories of ML tools do you use on a regular basis?  (Select all that apply) - Selected Choice - Other": "Do you use any automated machine learning tools (or partial AutoML tools) on a regular basis?  (Select all that apply) - Selected Choice - Other",
                        "Which of the following natural language processing (NLP) methods do you use on a regular basis?  (Select all that apply) - Selected Choice - Transformer language models (GPT-2, BERT, XLnet, etc)": "Which of the following natural language processing (NLP) methods do you use on a regular basis?  (Select all that apply) - Selected Choice - Transformer language models (GPT-3, BERT, XLnet, etc)",
                        "Which of the following cloud computing platforms do you use on a regular basis? (Select all that apply) - Selected Choice -  IBM Cloud ": "Which of the following cloud computing platforms do you use on a regular basis? (Select all that apply) - Selected Choice -  IBM Cloud / Red Hat ",
                        "Which automated machine learning tools (or partial AutoML tools) do you use on a regular basis?  (Select all that apply) - Selected Choice -  H20 Driverless AI  ": "Which automated machine learning tools (or partial AutoML tools) do you use on a regular basis?  (Select all that apply) - Selected Choice -  H2O Driverless AI  "}
                        , inplace=True)

# Merge options for the given columns into one column 
mcq_2020["Which of the following integrated development environments (IDE's) do you use on a regular basis?  (Select all that apply) - Selected Choice -  Visual Studio / Visual Studio Code "] = combine_multiple_option_columns(
    ["Which of the following integrated development environments (IDE's) do you use on a regular basis?  (Select all that apply) - Selected Choice -  Visual Studio / Visual Studio Code ",
    "Which of the following integrated development environments (IDE's) do you use on a regular basis?  (Select all that apply) - Selected Choice - Click to write Choice 13"],
    mcq_2020
)

# Merge options for the given columns into one column, and insert it into the data frame 
mcq_2020.insert(mcq_2020.shape[1], "Which of the following hosted notebook products do you use on a regular basis?  (Select all that apply) - Selected Choice - AWS Notebook Products (EMR Notebooks, Sagemaker Notebooks, etc) ",
                combine_multiple_option_columns(
                    ["Which of the following hosted notebook products do you use on a regular basis?  (Select all that apply) - Selected Choice -  Amazon Sagemaker Studio ",
                    "Which of the following hosted notebook products do you use on a regular basis?  (Select all that apply) - Selected Choice -  Amazon EMR Notebooks "],
                    mcq_2020
                ))

# Merge options for the given columns into one column, and insert it into the data frame
mcq_2020.insert(mcq_2020.shape[1], "Which of the following hosted notebook products do you use on a regular basis?  (Select all that apply) - Selected Choice -  Google Cloud Notebook Products (AI Platform, Datalab, etc) ",
                combine_multiple_option_columns(
                    ["Which of the following hosted notebook products do you use on a regular basis?  (Select all that apply) - Selected Choice - Google Cloud AI Platform Notebooks ",
                    "Which of the following hosted notebook products do you use on a regular basis?  (Select all that apply) - Selected Choice - Google Cloud Datalab Notebooks"],
                    mcq_2020
                ))

# Rename some of the columns to match with similar ones (i.e. ones asking the same question) in the other surveys
mcq_2020.rename(columns={"What data visualization libraries or tools do you use on a regular basis?  (Select all that apply) - Selected Choice -  D3 js ": "What data visualization libraries or tools do you use on a regular basis?  (Select all that apply) - Selected Choice -  D3.js ",
                        "Do you use any automated machine learning tools (or partial AutoML tools) on a regular basis?  (Select all that apply) - Selected Choice - Automation of full ML pipelines (e.g. Google AutoML, H20 Driverless AI)": "Do you use any automated machine learning tools (or partial AutoML tools) on a regular basis?  (Select all that apply) - Selected Choice - Automation of full ML pipelines (e.g. Google AutoML, H2O Driverless AI)",
                        "Do you use any of the following cloud computing products on a regular basis? (Select all that apply) - Selected Choice -  Amazon EC2 ": "Which specific cloud computing products do you use on a regular basis? (Select all that apply) - Selected Choice - AWS Elastic Compute Cloud (EC2)",
                        "Do you use any of the following cloud computing products on a regular basis? (Select all that apply) - Selected Choice -  Google Cloud Compute Engine ": "Which specific cloud computing products do you use on a regular basis? (Select all that apply) - Selected Choice - Google Compute Engine (GCE)",
                        "Do you use any of the following cloud computing products on a regular basis? (Select all that apply) - Selected Choice -  Azure Cloud Services ": "Which specific cloud computing products do you use on a regular basis? (Select all that apply) - Selected Choice - Azure Virtual Machines",
                        "Do you use any of the following cloud computing products on a regular basis? (Select all that apply) - Selected Choice -  Google Cloud Compute Engine ": "Which specific cloud computing products do you use on a regular basis? (Select all that apply) - Selected Choice - Google Compute Engine (GCE)",
                        "Do you use any of the following cloud computing products on a regular basis? (Select all that apply) - Selected Choice - No / None": "Which specific cloud computing products do you use on a regular basis? (Select all that apply) - Selected Choice - None",
                        "Do you use any of the following cloud computing products on a regular basis? (Select all that apply) - Selected Choice - Other": "Which specific cloud computing products do you use on a regular basis? (Select all that apply) - Selected Choice - Other",
                        "Which of the following big data products (relational databases, data warehouses, data lakes, or similar) do you use on a regular basis? (Select all that apply) - Selected Choice - MySQL ": "Which of the following relational database products do you use on a regular basis? (Select all that apply) - Selected Choice - MySQL",
                        "Which of the following big data products (relational databases, data warehouses, data lakes, or similar) do you use on a regular basis? (Select all that apply) - Selected Choice - PostgresSQL ": "Which of the following relational database products do you use on a regular basis? (Select all that apply) - Selected Choice - PostgresSQL",
                        "Which of the following big data products (relational databases, data warehouses, data lakes, or similar) do you use on a regular basis? (Select all that apply) - Selected Choice - SQLite ": "Which of the following relational database products do you use on a regular basis? (Select all that apply) - Selected Choice - SQLite",
                        "Which of the following big data products (relational databases, data warehouses, data lakes, or similar) do you use on a regular basis? (Select all that apply) - Selected Choice - Microsoft SQL Server ": "Which of the following relational database products do you use on a regular basis? (Select all that apply) - Selected Choice - Microsoft SQL Server",
                        "Which of the following big data products (relational databases, data warehouses, data lakes, or similar) do you use on a regular basis? (Select all that apply) - Selected Choice - Oracle Database ": "Which of the following relational database products do you use on a regular basis? (Select all that apply) - Selected Choice - Oracle Database",
                        "Which of the following big data products (relational databases, data warehouses, data lakes, or similar) do you use on a regular basis? (Select all that apply) - Selected Choice - Microsoft Access ": "Which of the following relational database products do you use on a regular basis? (Select all that apply) - Selected Choice - Microsoft Access",
                        "Which of the following big data products (relational databases, data warehouses, data lakes, or similar) do you use on a regular basis? (Select all that apply) - Selected Choice - Amazon DynamoDB ": "Which of the following relational database products do you use on a regular basis? (Select all that apply) - Selected Choice - AWS DynamoDB",
                        "Which of the following big data products (relational databases, data warehouses, data lakes, or similar) do you use on a regular basis? (Select all that apply) - Selected Choice - Microsoft Azure Data Lake Storage ": "Which of the following relational database products do you use on a regular basis? (Select all that apply) - Selected Choice - Azure SQL Database",
                        "Which of the following big data products (relational databases, data warehouses, data lakes, or similar) do you use on a regular basis? (Select all that apply) - Selected Choice - Google Cloud SQL ": "Which of the following relational database products do you use on a regular basis? (Select all that apply) - Selected Choice - Google Cloud SQL",
                        "Which of the following big data products (relational databases, data warehouses, data lakes, or similar) do you use on a regular basis? (Select all that apply) - Selected Choice - None": "Which of the following relational database products do you use on a regular basis? (Select all that apply) - Selected Choice - None",
                        "Which of the following big data products (relational databases, data warehouses, data lakes, or similar) do you use on a regular basis? (Select all that apply) - Selected Choice - Other": "Which of the following relational database products do you use on a regular basis? (Select all that apply) - Selected Choice - Other",
                        "Do you use any of the following machine learning products on a regular basis? (Select all that apply) - Selected Choice -  Amazon SageMaker ": "Which of the following machine learning products do you use on a regular basis? (Select all that apply) - Selected Choice - Amazon SageMaker",
                        "Do you use any of the following machine learning products on a regular basis? (Select all that apply) - Selected Choice -  Azure Machine Learning Studio ": "Which of the following machine learning products do you use on a regular basis? (Select all that apply) - Selected Choice - Azure Machine Learning Studio",
                        "Do you use any of the following machine learning products on a regular basis? (Select all that apply) - Selected Choice - No / None": "Which of the following machine learning products do you use on a regular basis? (Select all that apply) - Selected Choice - None",
                        "Do you use any of the following machine learning products on a regular basis? (Select all that apply) - Selected Choice - Other": "Which of the following machine learning products do you use on a regular basis? (Select all that apply) - Selected Choice - Other",
                        "Which of the following automated machine learning tools (or partial AutoML tools) do you use on a regular basis?  (Select all that apply) - Selected Choice -  Google Cloud AutoML ": "Which automated machine learning tools (or partial AutoML tools) do you use on a regular basis?  (Select all that apply) - Selected Choice -  Google AutoML ",
                        "Which of the following automated machine learning tools (or partial AutoML tools) do you use on a regular basis?  (Select all that apply) - Selected Choice -  H20 Driverless AI  ": "Which automated machine learning tools (or partial AutoML tools) do you use on a regular basis?  (Select all that apply) - Selected Choice -  H2O Driverless AI  ",
                        "Which of the following automated machine learning tools (or partial AutoML tools) do you use on a regular basis?  (Select all that apply) - Selected Choice -  Databricks AutoML ": "Which automated machine learning tools (or partial AutoML tools) do you use on a regular basis?  (Select all that apply) - Selected Choice -  Databricks AutoML ",
                        "Which of the following automated machine learning tools (or partial AutoML tools) do you use on a regular basis?  (Select all that apply) - Selected Choice -  DataRobot AutoML ": "Which automated machine learning tools (or partial AutoML tools) do you use on a regular basis?  (Select all that apply) - Selected Choice -  DataRobot AutoML ",
                        "Which of the following automated machine learning tools (or partial AutoML tools) do you use on a regular basis?  (Select all that apply) - Selected Choice - No / None": "Which automated machine learning tools (or partial AutoML tools) do you use on a regular basis?  (Select all that apply) - Selected Choice - None",
                        "Which of the following automated machine learning tools (or partial AutoML tools) do you use on a regular basis?  (Select all that apply) - Selected Choice - Other": "Which automated machine learning tools (or partial AutoML tools) do you use on a regular basis?  (Select all that apply) - Selected Choice - Other"},
                        inplace=True)

# Merge options for the given columns into one column, and insert it into the data frame
mcq_2021.insert(mcq_2021.shape[1], "Which of the following integrated development environments (IDE's) do you use on a regular basis?  (Select all that apply) - Selected Choice -  Visual Studio / Visual Studio Code ",
                combine_multiple_option_columns(
                    ["Which of the following integrated development environments (IDE's) do you use on a regular basis?  (Select all that apply) - Selected Choice -  Visual Studio ",
                    "Which of the following integrated development environments (IDE's) do you use on a regular basis?  (Select all that apply) - Selected Choice -  Visual Studio Code (VSCode) "],
                    mcq_2021
                ))

# Merge options for the given columns into one column, and insert it into the data frame
mcq_2021.insert(mcq_2021.shape[1], "Which of the following hosted notebook products do you use on a regular basis?  (Select all that apply) - Selected Choice - AWS Notebook Products (EMR Notebooks, Sagemaker Notebooks, etc) ",
                combine_multiple_option_columns(
                    ["Which of the following hosted notebook products do you use on a regular basis?  (Select all that apply) - Selected Choice -  Amazon Sagemaker Studio Notebooks ",
                    "Which of the following hosted notebook products do you use on a regular basis?  (Select all that apply) - Selected Choice -  Amazon EMR Notebooks "],
                    mcq_2021
                ))

# Merge options for the given columns into one column, and insert it into the data frame
mcq_2021.insert(mcq_2021.shape[1], "Which of the following hosted notebook products do you use on a regular basis?  (Select all that apply) - Selected Choice -  Google Cloud Notebook Products (AI Platform, Datalab, etc) ",
                combine_multiple_option_columns(
                    ["Which of the following hosted notebook products do you use on a regular basis?  (Select all that apply) - Selected Choice - Google Cloud Notebooks (AI Platform / Vertex AI) ",
                    "Which of the following hosted notebook products do you use on a regular basis?  (Select all that apply) - Selected Choice - Google Cloud Datalab"],
                    mcq_2021
                ))

# Rename some of the columns to match with similar ones (i.e. ones asking the same question) in the other surveys
mcq_2021.rename(columns={"What data visualization libraries or tools do you use on a regular basis?  (Select all that apply) - Selected Choice -  D3 js ": "What data visualization libraries or tools do you use on a regular basis?  (Select all that apply) - Selected Choice -  D3.js ",
                        "Which types of specialized hardware do you use on a regular basis?  (Select all that apply) - Selected Choice -  NVIDIA GPUs ": "Which types of specialized hardware do you use on a regular basis?  (Select all that apply) - Selected Choice - GPUs",
                        "Which types of specialized hardware do you use on a regular basis?  (Select all that apply) - Selected Choice -  Google Cloud TPUs ": "Which types of specialized hardware do you use on a regular basis?  (Select all that apply) - Selected Choice - TPUs",
                        "Do you use any of the following cloud computing products on a regular basis? (Select all that apply) - Selected Choice -  Amazon Elastic Compute Cloud (EC2) ": "Which specific cloud computing products do you use on a regular basis? (Select all that apply) - Selected Choice - AWS Elastic Compute Cloud (EC2)",
                        "Do you use any of the following cloud computing products on a regular basis? (Select all that apply) - Selected Choice -  Microsoft Azure Virtual Machines ": "Which specific cloud computing products do you use on a regular basis? (Select all that apply) - Selected Choice - Azure Virtual Machines",
                        "Do you use any of the following cloud computing products on a regular basis? (Select all that apply) - Selected Choice -  Google Cloud Compute Engine ": "Which specific cloud computing products do you use on a regular basis? (Select all that apply) - Selected Choice - Google Compute Engine (GCE)",
                        "Do you use any of the following cloud computing products on a regular basis? (Select all that apply) - Selected Choice - No / None": "Which specific cloud computing products do you use on a regular basis? (Select all that apply) - Selected Choice - None",
                        "Do you use any of the following cloud computing products on a regular basis? (Select all that apply) - Selected Choice - Other": "Which specific cloud computing products do you use on a regular basis? (Select all that apply) - Selected Choice - Other",
                        "Which of the following big data products (relational databases, data warehouses, data lakes, or similar) do you use on a regular basis? (Select all that apply) - Selected Choice - MySQL ": "Which of the following relational database products do you use on a regular basis? (Select all that apply) - Selected Choice - MySQL",
                        "Which of the following big data products (relational databases, data warehouses, data lakes, or similar) do you use on a regular basis? (Select all that apply) - Selected Choice - PostgreSQL ": "Which of the following relational database products do you use on a regular basis? (Select all that apply) - Selected Choice - PostgresSQL",
                        "Which of the following big data products (relational databases, data warehouses, data lakes, or similar) do you use on a regular basis? (Select all that apply) - Selected Choice - SQLite ": "Which of the following relational database products do you use on a regular basis? (Select all that apply) - Selected Choice - SQLite",
                        "Which of the following big data products (relational databases, data warehouses, data lakes, or similar) do you use on a regular basis? (Select all that apply) - Selected Choice - Microsoft SQL Server ": "Which of the following relational database products do you use on a regular basis? (Select all that apply) - Selected Choice - Microsoft SQL Server",
                        "Which of the following big data products (relational databases, data warehouses, data lakes, or similar) do you use on a regular basis? (Select all that apply) - Selected Choice - Oracle Database ": "Which of the following relational database products do you use on a regular basis? (Select all that apply) - Selected Choice - Oracle Database",
                        "Which of the following big data products (relational databases, data warehouses, data lakes, or similar) do you use on a regular basis? (Select all that apply) - Selected Choice - Microsoft Access ": "Which of the following relational database products do you use on a regular basis? (Select all that apply) - Selected Choice - Microsoft Access",
                        "Which of the following big data products (relational databases, data warehouses, data lakes, or similar) do you use on a regular basis? (Select all that apply) - Selected Choice - Amazon DynamoDB ": "Which of the following relational database products do you use on a regular basis? (Select all that apply) - Selected Choice - AWS DynamoDB",
                        "Which of the following big data products (relational databases, data warehouses, data lakes, or similar) do you use on a regular basis? (Select all that apply) - Selected Choice - Microsoft Azure SQL Database ": "Which of the following relational database products do you use on a regular basis? (Select all that apply) - Selected Choice - Azure SQL Database",
                        "Which of the following big data products (relational databases, data warehouses, data lakes, or similar) do you use on a regular basis? (Select all that apply) - Selected Choice - Google Cloud SQL ": "Which of the following relational database products do you use on a regular basis? (Select all that apply) - Selected Choice - Google Cloud SQL",
                        "Which of the following big data products (relational databases, data warehouses, data lakes, or similar) do you use on a regular basis? (Select all that apply) - Selected Choice - None": "Which of the following relational database products do you use on a regular basis? (Select all that apply) - Selected Choice - None",
                        "Which of the following big data products (relational databases, data warehouses, data lakes, or similar) do you use on a regular basis? (Select all that apply) - Selected Choice - Other": "Which of the following relational database products do you use on a regular basis? (Select all that apply) - Selected Choice - Other",
                        "Do you use any of the following managed machine learning products on a regular basis? (Select all that apply) - Selected Choice -  Amazon SageMaker ": "Which of the following machine learning products do you use on a regular basis? (Select all that apply) - Selected Choice - Amazon SageMaker",
                        "Do you use any of the following managed machine learning products on a regular basis? (Select all that apply) - Selected Choice -  Azure Machine Learning Studio ": "Which of the following machine learning products do you use on a regular basis? (Select all that apply) - Selected Choice - Azure Machine Learning Studio",
                        "Do you use any of the following managed machine learning products on a regular basis? (Select all that apply) - Selected Choice - No / None": "Which of the following machine learning products do you use on a regular basis? (Select all that apply) - Selected Choice - None",
                        "Do you use any of the following managed machine learning products on a regular basis? (Select all that apply) - Selected Choice - Other": "Which of the following machine learning products do you use on a regular basis? (Select all that apply) - Selected Choice - Other",
                        "Which of the following automated machine learning tools (or partial AutoML tools) do you use on a regular basis?  (Select all that apply) - Selected Choice -  Google Cloud AutoML ": "Which automated machine learning tools (or partial AutoML tools) do you use on a regular basis?  (Select all that apply) - Selected Choice -  Google AutoML ",
                        "Which of the following automated machine learning tools (or partial AutoML tools) do you use on a regular basis?  (Select all that apply) - Selected Choice -  H2O Driverless AI  ": "Which automated machine learning tools (or partial AutoML tools) do you use on a regular basis?  (Select all that apply) - Selected Choice -  H2O Driverless AI  ",
                        "Which of the following automated machine learning tools (or partial AutoML tools) do you use on a regular basis?  (Select all that apply) - Selected Choice -  Databricks AutoML ": "Which automated machine learning tools (or partial AutoML tools) do you use on a regular basis?  (Select all that apply) - Selected Choice -  Databricks AutoML ",
                        "Which of the following automated machine learning tools (or partial AutoML tools) do you use on a regular basis?  (Select all that apply) - Selected Choice -  DataRobot AutoML ": "Which automated machine learning tools (or partial AutoML tools) do you use on a regular basis?  (Select all that apply) - Selected Choice -  DataRobot AutoML ",
                        "Which of the following automated machine learning tools (or partial AutoML tools) do you use on a regular basis?  (Select all that apply) - Selected Choice - No / None": "Which automated machine learning tools (or partial AutoML tools) do you use on a regular basis?  (Select all that apply) - Selected Choice - None",
                        "Which of the following automated machine learning tools (or partial AutoML tools) do you use on a regular basis?  (Select all that apply) - Selected Choice - Other": "Which automated machine learning tools (or partial AutoML tools) do you use on a regular basis?  (Select all that apply) - Selected Choice - Other"},
                        inplace=True)

# Insert year column at beginning of each data frame
mcq_2019.insert(0, "Year", 2019) 
mcq_2020.insert(0, "Year", 2020)
mcq_2021.insert(0, "Year", 2021)

# Take intersection of data frames with respect to columns
df = pd.concat([mcq_2019, mcq_2020, mcq_2021], join="inner", axis=0)

# Set new data frame index to be simply the numbers 1 to n, where n is the number of rows
df.index = range(1, df.shape[0] + 1)

# Save the data frame to a new .csv file
df.to_csv("Kaggle_survey 2019-2021.csv")