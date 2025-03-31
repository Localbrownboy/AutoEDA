import pandas as pd
import numpy as np
from openai import Client
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tool_list import viz_tools_info, trans_tools_info
from IPython.display import display, Markdown  

client = Client()

def _analyze_transformation(prompt, history=None):
    """
    Send the transformation prompt (with conversation history) to the LLM
    and return its analysis and suggestions.
    """
    if history is not None:
        history.add_message("user", prompt)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=history.get_history(),
            temperature=0.7
        )
        llm_output = response.choices[0].message.content.strip()
        history.add_message("assistant", llm_output)
    else:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        llm_output = response.choices[0].message.content.strip()
    return llm_output

def fill_missing(df, column, strategy="mean", title="Missing Value Imputation", history=None):
    df_transformed = df.copy()
    missing_before = df_transformed[column].isna().sum()
    
    if strategy == "mean":
        fill_value = df_transformed[column].mean()
    elif strategy == "median":
        fill_value = df_transformed[column].median()
    elif strategy == "mode":
        fill_value = df_transformed[column].mode()[0]
    else:
        fill_value = strategy
    
    df_transformed[column] = df_transformed[column].fillna(fill_value)
    missing_after = df_transformed[column].isna().sum()
    
    summary_text = (
        f"Transformation: {title}\n"
        f"Column: {column}\n"
        f"Strategy: {strategy}\n"
        f"Missing values before: {missing_before}\n"
        f"Missing values after: {missing_after}\n"
        f"First 5 values after transformation: {df_transformed[column].head().to_list()}\n\n"
        "Please analyze this transformation, provide insights on the applied changes, and offer suggestions for further data transformations or improvements. "
        "Provide the exact function call to make from the list below. "
        "Format your response in markdown with sections titled 'Analysis' and 'Suggestions'.\n"
        f"{viz_tools_info}\n{trans_tools_info}"
    )
    
    analysis = _analyze_transformation(summary_text, history=history)
    display(Markdown("---"))
    display(Markdown("### LLM Analysis & Suggestion"))
    display(Markdown(analysis))
    return df_transformed

def scale_data(df, columns, method="minmax", title="Data Scaling", history=None):
    df_transformed = df.copy()
    
    scaler = MinMaxScaler() if method == "minmax" else StandardScaler()
    original_values = {col: df_transformed[col].head().to_list() for col in columns}
    df_transformed[columns] = scaler.fit_transform(df_transformed[columns])
    scaled_values = {col: df_transformed[col].head().to_list() for col in columns}
    
    summary_text = (
        f"Transformation: {title}\n"
        f"Columns: {columns}\n"
        f"Method: {method}\n"
        f"Original first 5 values: {original_values}\n"
        f"Scaled first 5 values: {scaled_values}\n\n"
        "Please analyze this transformation, provide insights on the applied changes, and offer suggestions for further data transformations or improvements. "
        "Provide the exact function call to make from the list below. "
        "Format your response in markdown with sections titled 'Analysis' and 'Suggestions'.\n"
        f"{viz_tools_info}\n{trans_tools_info}"
    )
    
    analysis = _analyze_transformation(summary_text, history=history)
    display(Markdown("---"))
    display(Markdown("### LLM Analysis & Suggestion"))
    display(Markdown(analysis))
    return df_transformed

def encode_categorical(df, column, drop_first=True, title="Categorical Encoding", history=None):
    df_transformed = df.copy()
    original_values = df_transformed[column].unique().tolist()
    df_encoded = pd.get_dummies(df_transformed, columns=[column], drop_first=drop_first)
    new_columns = [col for col in df_encoded.columns if col.startswith(column + "_")]
    
    summary_text = (
        f"Transformation: {title}\n"
        f"Original column: {column}\n"
        f"Original unique values: {original_values}\n"
        f"New columns created: {new_columns}\n"
        f"Head of new columns: {df_encoded[new_columns].head().to_dict()}\n\n"
        "Please analyze this transformation, provide insights on the applied changes, and offer suggestions for further data transformations or improvements. "
        "Provide the exact function call to make from the list below. "
        "Format your response in markdown with sections titled 'Analysis' and 'Suggestions'.\n"
        f"{viz_tools_info}\n{trans_tools_info}"
    )
    
    analysis = _analyze_transformation(summary_text, history=history)
    display(Markdown("---"))
    display(Markdown("### LLM Analysis & Suggestion"))
    display(Markdown(analysis))
    return df_encoded

def log_transform(df, column, title="Log Transformation", history=None):
    df_transformed = df.copy()
    if (df_transformed[column] <= 0).any():
        df_transformed[column] = np.log1p(df_transformed[column])
        method_used = "np.log1p"
    else:
        df_transformed[column] = np.log(df_transformed[column])
        method_used = "np.log"
        
    summary_text = (
        f"Transformation: {title}\n"
        f"Column: {column}\n"
        f"Method used: {method_used}\n"
        f"First 5 values after transformation: {df_transformed[column].head().to_list()}\n\n"
        "Please analyze this transformation, provide insights on the applied changes, and offer suggestions for further data transformations or improvements. "
        "Provide the exact function call to make from the list below. "
        "Format your response in markdown with sections titled 'Analysis' and 'Suggestions'.\n"
        f"{viz_tools_info}\n{trans_tools_info}"
    )
    
    analysis = _analyze_transformation(summary_text, history=history)
    display(Markdown("---"))
    display(Markdown("### LLM Analysis & Suggestion"))
    display(Markdown(analysis))
    return df_transformed

def power_transform(df, column, power=2, title="Power Transformation", history=None):
    df_transformed = df.copy()
    original_values = df_transformed[column].head().to_list()
    df_transformed[column] = df_transformed[column] ** power
    transformed_values = df_transformed[column].head().to_list()
    
    summary_text = (
        f"Transformation: {title}\n"
        f"Column: {column}\n"
        f"Power: {power}\n"
        f"Original first 5 values: {original_values}\n"
        f"Transformed first 5 values: {transformed_values}\n\n"
        "Please analyze this transformation, provide insights on the applied changes, and offer suggestions for further data transformations or improvements. "
        "Provide the exact function call to make from the list below. "
        "Format your response in markdown with sections titled 'Analysis' and 'Suggestions'.\n"
        f"{viz_tools_info}\n{trans_tools_info}"
    )
    
    analysis = _analyze_transformation(summary_text, history=history)
    display(Markdown("---"))
    display(Markdown("### LLM Analysis & Suggestion"))
    display(Markdown(analysis))
    return df_transformed

def bin_continuous(df, column, bins, labels=None, title="Binning Continuous Variable", history=None):
    df_transformed = df.copy()
    new_column = f"{column}_binned"
    df_transformed[new_column] = pd.cut(df_transformed[column], bins=bins, labels=labels, include_lowest=True)
    
    summary_text = (
        f"Transformation: {title}\n"
        f"Original Column: {column}\n"
        f"New Binned Column: {new_column}\n"
        f"Bins: {bins}\n"
        f"Sample binned values: {df_transformed[new_column].head().to_list()}\n\n"
        "Please analyze this transformation, provide insights on the applied changes, and offer suggestions for further data transformations or improvements. "
        "Provide the exact function call to make from the list below. "
        "Format your response in markdown with sections titled 'Analysis' and 'Suggestions'.\n"
        f"{viz_tools_info}\n{trans_tools_info}"
    )
    
    analysis = _analyze_transformation(summary_text, history=history)
    display(Markdown("---"))
    display(Markdown("### LLM Analysis & Suggestion"))
    display(Markdown(analysis))
    return df_transformed
