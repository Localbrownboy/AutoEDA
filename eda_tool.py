from openai import Client
import pandas as pd
from IPython.display import display, Markdown
import json
from history_manager import HistoryManager
from tool_list import viz_tools_info, trans_tools_info
# Import the visualization module
from visualizations import (
    plot_box,
    plot_scatter,
    plot_histogram,
    plot_violin,
    plot_pair,
    plot_heatmap
)

# Import transformation module 
from transformations import (
    fill_missing, 
    scale_data, 
    encode_categorical, 
    log_transform, 
    power_transform, 
    bin_continuous
)




# (api key loaded from env)
client = Client()

class EDA:
    def __init__(self, filepath):
        """Initialize the EDA tool with a dataset file path and conversation history."""
        self.data = pd.read_csv(filepath)
        self.history = HistoryManager() 
        self.summary = self._get_data_summary()
        
        # Get LLM response (summary and plot suggestions) including available tool details
        llm_response = self._get_llm_response()
        
        # Display dataset summary and visualization suggestions
        display(Markdown("### Data Summary"))
        display(Markdown(llm_response['summary']))
        display(Markdown("### Suggested Plots"))
        for suggestion in llm_response['suggestions']:
            display(Markdown(f"- {suggestion}"))
    
    def _get_data_summary(self):
        """Compute basic statistics about the dataset."""
        summary = {
            'n_rows': len(self.data),
            'n_cols': len(self.data.columns),
            'columns': [],
            'missing_values': self.data.isnull().sum().to_dict()
        }
        for col in self.data.columns:
            col_info = {'name': col, 'dtype': str(self.data[col].dtype)}
            if pd.api.types.is_numeric_dtype(self.data[col]):
                col_info.update({
                    'min': self.data[col].min(),
                    'max': self.data[col].max(),
                    'mean': self.data[col].mean(),
                    'median': self.data[col].median(),
                    'std': self.data[col].std()
                })
            elif pd.api.types.is_object_dtype(self.data[col]):
                col_info.update({
                    'n_unique': self.data[col].nunique(),
                    'top_values': self.data[col].value_counts().head(5).to_dict()
                })
            summary['columns'].append(col_info)
        return summary

    def _get_llm_response(self):
        """
        Generate summary and plot suggestions using the LLM.
        This prompt includes details about available visualization and transformation tools.
        """
        prompt = f"I have a dataset with {self.summary['n_rows']} rows and {self.summary['n_cols']} columns.\n\nColumns:\n"
        for col in self.summary['columns']:
            prompt += f"- {col['name']} ({col['dtype']})"
            if 'min' in col:
                prompt += f": min={col['min']}, max={col['max']}, mean={col['mean']:.2f}, median={col['median']:.2f}, std={col['std']:.2f}"
            elif 'n_unique' in col:
                prompt += f": {col['n_unique']} unique values, top values: {col['top_values']}"
            prompt += "\n"
        prompt += f"\nMissing values: {self.summary['missing_values']}\n\n"
        prompt += (
            "Provide a concise summary of the dataset, highlighting key characteristics such as size, variable types, and missing values. "
            "Then, suggest 3-5 interesting plots for further exploration. "
            "For each plot suggestion, include the exact function call from the visualization tool (from the list below) that can be used to generate the graph. "
            "Return the response in JSON format with keys 'summary' (string) and 'suggestions' (list of strings).\n\n"
        )
        prompt += viz_tools_info + "\n" + trans_tools_info

        self.history.add_message("user", prompt)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=self.history.get_history(),
            temperature=0.7
        )
        llm_output = response.choices[0].message.content.strip()
        self.history.add_message("assistant", llm_output)
        
        # Strip markdown formatting if present
        if llm_output.startswith("```"):
            llm_output = llm_output.strip("`").strip()
            if llm_output.lower().startswith("json"):
                llm_output = llm_output[len("json"):].strip()
        return json.loads(llm_output)
    
    def get_transformation_suggestions(self):
        """
        Generate transformation suggestions using the LLM based on the dataset summary.
        Returns a JSON response.
        """
        prompt = f"Based on the following dataset summary, suggest 3-5 data transformation operations to improve data quality or reveal insights.\n\nDataset Summary:\nRows: {self.summary['n_rows']}\nColumns: {self.summary['n_cols']}\n"
        prompt += "Columns:\n"
        for col in self.summary['columns']:
            prompt += f"- {col['name']} ({col['dtype']})"
            if 'min' in col:
                prompt += f": min={col['min']}, max={col['max']}, mean={col['mean']:.2f}, median={col['median']:.2f}, std={col['std']:.2f}\n"
            elif 'n_unique' in col:
                prompt += f": {col['n_unique']} unique values\n"
            else:
                prompt += "\n"
        prompt += f"\nMissing values: {self.summary['missing_values']}\n\n"
        prompt += trans_tools_info
        prompt += "\nReturn the response in JSON format with keys 'summary' (string) and 'suggestions' (list of strings)."
        
        self.history.add_message("user", prompt)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=self.history.get_history(),
            temperature=0.7
        )
        llm_output = response.choices[0].message.content.strip()
        self.history.add_message("assistant", llm_output)
        
        if llm_output.startswith("```"):
            llm_output = llm_output.strip("`").strip()
            if llm_output.lower().startswith("json"):
                llm_output = llm_output[len("json"):].strip()
        return json.loads(llm_output)
    
    def suggest_transformations(self):
        """Display transformation suggestions returned by the LLM."""
        suggestions = self.get_transformation_suggestions()
        display(Markdown("### Transformation Suggestions"))
        display(Markdown(suggestions['summary']))
        for suggestion in suggestions['suggestions']:
            display(Markdown(f"- {suggestion}"))
    
    def plot_box(self, x, y, hue=None, title="Box Plot"):
        plot_box(self.data, x, y, hue, title, history=self.history)

    def plot_scatter(self, x, y, hue=None, title="Scatter Plot"):
        plot_scatter(self.data, x, y, hue, title, history=self.history)

    def plot_histogram(self, column, bins=10, title="Histogram"):
        plot_histogram(self.data, column, bins, title, history=self.history)

    def plot_violin(self, x, y, title="Violin Plot"):
        plot_violin(self.data, x, y, title, history=self.history)

    def plot_pair(self, columns, title="Pair Plot"):
        plot_pair(self.data, columns, title, history=self.history)

    def plot_heatmap(self, title="Heatmap"):
        plot_heatmap(self.data, title, history=self.history)

    def fill_missing(self, column, strategy="mean", title="Missing Value Imputation"):
        self.data = fill_missing(self.data, column, strategy, title, history=self.history)
        return self.data

    def scale_data(self, columns, method="minmax", title="Data Scaling"):
        self.data = scale_data(self.data, columns, method, title, history=self.history)
        return self.data

    def encode_categorical(self, column, drop_first=True, title="Categorical Encoding"):
        self.data = encode_categorical(self.data, column, drop_first, title, history=self.history)
        return self.data

    def log_transform(self, column, title="Log Transformation"):
        self.data = log_transform(self.data, column, title, history=self.history)
        return self.data

    def power_transform(self, column, power=2, title="Power Transformation"):
        self.data = power_transform(self.data, column, power, title, history=self.history)
        return self.data

    def bin_continuous(self, column, bins, labels=None, title="Binning Continuous Variable"):
        self.data = bin_continuous(self.data, column, bins, labels, title, history=self.history)
        return self.data
