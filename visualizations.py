import matplotlib.pyplot as plt
import seaborn as sns
import re
from openai import Client
import base64
from IPython.display import display, Markdown  
from history_manager import HistoryManager 
from tool_list import viz_tools_info, trans_tools_info

client = Client()

def _capture_plot_as_base64():
    import io
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return image_base64

def _analyze_plot(text_prompt, image_base64=None):
    """
    Send the prompt along with the plot image (if provided) to the LLM.
    The message content is structured as a list containing both text and image parts.
    """
    if image_base64:
        content = [
            {"type": "text", "text": text_prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
        ]
    else:
        content = [
            {"type": "text", "text": text_prompt}
        ]
    
    response = client.chat.completions.create(
        model="gpt-4o-mini", 
        messages=[{"role": "user", "content": content}],
        temperature=0.7
    )
    return response.choices[0].message.content.strip()

def _process_plot_analysis(title, additional_info, history=None):
    image_base64 = _capture_plot_as_base64()
    plt.show()
    plt.close()
    
    # Updated prompt to request markdown formatting
    text_prompt = (
        f"I have generated a plot titled '{title}'. {additional_info}\n\n"
        "Please analyze this plot, provide insights on the observed patterns, and offer suggestions for further visualizations or data transformations. Provide the exact function call to make from the list below."
        "Format your response in markdown with sections titled 'Analysis' and 'Suggestions'."
    )
    text_prompt += viz_tools_info + "\n" + trans_tools_info
    
    analysis = _analyze_plot(text_prompt, image_base64)
    
    # If conversation history is provided, update it using the history manager
    if history is not None:
        stripped_prompt = re.sub(
            r"(data:image/jpeg;base64,)[A-Za-z0-9+/=]+",
            "data:image/jpeg;base64,<IMAGE_DATA>",
            text_prompt
        )
        history.add_message("user", stripped_prompt)
        history.add_message("assistant", analysis)
    
    # Display the analysis with markdown formatting instead of printing
    display(Markdown("---"))  # Separator for visual clarity
    display(Markdown("### LLM Analysis & Suggestion"))
    display(Markdown(analysis))

def plot_box(df, x, y, hue=None, title="Box Plot", history=None):
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, x=x, y=y, hue=hue)
    plt.title(title)
    plt.tight_layout()
    
    additional_info = f"This box plot visualizes the distribution of '{y}' grouped by '{x}'."
    if hue:
        additional_info += f" A hue based on '{hue}' is applied."
    _process_plot_analysis(title, additional_info, history)

def plot_scatter(df, x, y, hue=None, title="Scatter Plot", history=None):
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x=x, y=y, hue=hue)
    plt.title(title)
    plt.tight_layout()
    
    additional_info = f"This scatter plot visualizes the relationship between '{x}' and '{y}'."
    if hue:
        additional_info += f" Points are colored by '{hue}'."
    _process_plot_analysis(title, additional_info, history)

def plot_histogram(df, column, bins=10, title="Histogram", history=None):
    plt.figure(figsize=(8, 6))
    plt.hist(df[column].dropna(), bins=bins, edgecolor="black")
    plt.title(title)
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.tight_layout()
    
    additional_info = f"This histogram shows the distribution of the variable '{column}' using {bins} bins."
    _process_plot_analysis(title, additional_info, history)

def plot_violin(df, x, y, title="Violin Plot", history=None):
    plt.figure(figsize=(8, 6))
    sns.violinplot(data=df, x=x, y=y)
    plt.title(title)
    plt.tight_layout()
    
    additional_info = f"This violin plot compares the distribution of '{y}' across categories of '{x}'."
    _process_plot_analysis(title, additional_info, history)

def plot_pair(df, columns, title="Pair Plot", history=None):
    pairplot_fig = sns.pairplot(df[columns].dropna())
    pairplot_fig.fig.suptitle(title)
    pairplot_fig.fig.tight_layout()
    
    additional_info = f"This pair plot shows pairwise relationships among the variables: {', '.join(columns)}."
    # Updated prompt to request markdown formatting and simplified
    prompt = (
        f"I have generated a pair plot titled '{title}'. {additional_info}\n\n"
        "Please analyze the pair plot, provide insights on the relationships between the variables, and suggest any improvements, further visualizations, or data transformations. "
        "Format your response in markdown with sections titled 'Analysis' and 'Suggestions'."
    )
    analysis = _analyze_plot(prompt)
    if history is not None:
        history.add_message("user", prompt)
        history.add_message("assistant", analysis)
    
    # Display the analysis with markdown formatting instead of printing
    display(Markdown("---"))  # Separator for visual clarity
    display(Markdown("### LLM Analysis & Suggestion"))
    display(Markdown(analysis))
    
    plt.show()
    plt.close()

def plot_heatmap(df, title="Heatmap", history=None):
    plt.figure(figsize=(10, 8))
    corr = df.select_dtypes(include="number").corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title(title)
    plt.tight_layout()
    
    additional_info = "This heatmap represents the correlation matrix of the numerical variables in the dataset."
    _process_plot_analysis(title, additional_info, history)