viz_tools_info = (
            "Available visualization methods (use these methods of the EDA class):\n"
            "1. plot_box(x, y, hue=None, title='Box Plot')\n"
            "2. plot_scatter(x, y, hue=None, title='Scatter Plot')\n"
            "3. plot_histogram(column, bins=10, title='Histogram')\n"
            "4. plot_violin(x, y, title='Violin Plot')\n"
            "5. plot_pair(columns, title='Pair Plot')\n"
            "6. plot_heatmap(title='Heatmap')\n"
        )
trans_tools_info = (
            "Available transformation methods (use these methods of the EDA class):\n"
            "1. fill_missing(column, strategy='mean', title='Missing Value Imputation')\n"
            "2. scale_data(columns, method='minmax', title='Data Scaling')\n"
            "3. encode_categorical(column, drop_first=True, title='Categorical Encoding')\n"
            "4. log_transform(column, title='Log Transformation')\n"
            "5. power_transform(column, power=2, title='Power Transformation')\n"
            "6. bin_continuous(column, bins, labels=None, title='Binning Continuous Variable')\n"
        )