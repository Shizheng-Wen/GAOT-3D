import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
import os

def plot_num_points_distribution(csv_path, output_dir=None, show_plot=True):
    """
    Plot the distribution of number of points from drivaerml statistics CSV
    
    Parameters:
    -----------
    csv_path : str
        Path to the CSV file containing statistics
    output_dir : str, optional
        Directory to save the plots (default: same directory as CSV)
    show_plot : bool
        Whether to display the plots (default: True)
    
    Returns:
    --------
    pd.DataFrame : The loaded statistics data
    """
    
    # Read CSV file
    if not os.path.exists(csv_path):
        print(f"Error: CSV file {csv_path} does not exist")
        return None
    
    df = pd.read_csv(csv_path)
    
    if 'num_points' not in df.columns:
        print("Error: 'num_points' column not found in CSV file")
        return None
    
    print(f"Loaded statistics for {len(df)} files")
    
    # Set up output directory
    if output_dir is None:
        output_dir = os.path.dirname(csv_path)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Set style for better looking plots
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('DrivaerML Dataset: Number of Points Distribution Analysis', fontsize=16, fontweight='bold')
    
    # 1. Histogram
    ax1 = axes[0, 0]
    num_points = df['num_points']
    
    # Calculate bins using Freedman-Diaconis rule
    q75, q25 = np.percentile(num_points, [75, 25])
    iqr = q75 - q25
    bin_width = 2 * iqr / (len(num_points) ** (1/3))
    bins = int((num_points.max() - num_points.min()) / bin_width) if bin_width > 0 else 30
    bins = min(max(bins, 10), 50)  # Limit between 10 and 50 bins
    
    ax1.hist(num_points, bins=bins, alpha=0.7, color='skyblue', edgecolor='black', linewidth=0.5)
    ax1.set_xlabel('Number of Points')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Histogram of Number of Points')
    ax1.grid(True, alpha=0.3)
    
    # Add statistics text
    stats_text = f'Mean: {num_points.mean():.0f}\nMedian: {num_points.median():.0f}\nStd: {num_points.std():.0f}'
    ax1.text(0.98, 0.98, stats_text, transform=ax1.transAxes, 
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 2. Box plot
    ax2 = axes[0, 1]
    box_plot = ax2.boxplot(num_points, patch_artist=True)
    box_plot['boxes'][0].set_facecolor('lightcoral')
    ax2.set_ylabel('Number of Points')
    ax2.set_title('Box Plot of Number of Points')
    ax2.grid(True, alpha=0.3)
    
    # Add outlier information
    Q1 = num_points.quantile(0.25)
    Q3 = num_points.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = num_points[(num_points < lower_bound) | (num_points > upper_bound)]
    
    outlier_text = f'Outliers: {len(outliers)}\nIQR: {IQR:.0f}'
    ax2.text(0.02, 0.98, outlier_text, transform=ax2.transAxes,
             verticalalignment='top', horizontalalignment='left',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 3. Cumulative distribution
    ax3 = axes[1, 0]
    sorted_points = np.sort(num_points)
    cumulative_prob = np.arange(1, len(sorted_points) + 1) / len(sorted_points)
    ax3.plot(sorted_points, cumulative_prob, linewidth=2, color='green')
    ax3.set_xlabel('Number of Points')
    ax3.set_ylabel('Cumulative Probability')
    ax3.set_title('Cumulative Distribution Function (CDF)')
    ax3.grid(True, alpha=0.3)
    
    # Add percentile lines
    percentiles = [25, 50, 75, 90, 95]
    for p in percentiles:
        val = np.percentile(num_points, p)
        ax3.axvline(val, color='red', linestyle='--', alpha=0.7)
        ax3.text(val, p/100, f'P{p}: {val:.0f}', rotation=90, 
                verticalalignment='bottom', horizontalalignment='right')
    
    # 4. Summary statistics table
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Calculate detailed statistics
    stats_data = {
        'Statistic': ['Count', 'Mean', 'Median', 'Mode', 'Std Dev', 'Min', 'Max', 
                     'Range', 'Q1 (25%)', 'Q3 (75%)', 'IQR', 'Skewness', 'Kurtosis'],
        'Value': [
            f'{len(num_points)}',
            f'{num_points.mean():.1f}',
            f'{num_points.median():.1f}',
            f'{num_points.mode().iloc[0]:.0f}',
            f'{num_points.std():.1f}',
            f'{num_points.min():.0f}',
            f'{num_points.max():.0f}',
            f'{num_points.max() - num_points.min():.0f}',
            f'{num_points.quantile(0.25):.1f}',
            f'{num_points.quantile(0.75):.1f}',
            f'{IQR:.1f}',
            f'{num_points.skew():.3f}',
            f'{num_points.kurtosis():.3f}'
        ]
    }
    
    # Create table
    table = ax4.table(cellText=[[stat, val] for stat, val in zip(stats_data['Statistic'], stats_data['Value'])],
                     colLabels=['Statistic', 'Value'],
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.6, 0.4])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style the table
    for i in range(len(stats_data['Statistic']) + 1):
        if i == 0:  # Header
            table[(i, 0)].set_facecolor('#4CAF50')
            table[(i, 1)].set_facecolor('#4CAF50')
            table[(i, 0)].set_text_props(weight='bold', color='white')
            table[(i, 1)].set_text_props(weight='bold', color='white')
        else:
            if i % 2 == 0:
                table[(i, 0)].set_facecolor('#f0f0f0')
                table[(i, 1)].set_facecolor('#f0f0f0')
    
    ax4.set_title('Statistical Summary', fontweight='bold', pad=20)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(output_dir, 'num_points_distribution.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Distribution plot saved to: {output_path}")
    
    # Show plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    # Create additional detailed histogram
    plt.figure(figsize=(12, 8))
    
    # Plot histogram with density curve
    plt.hist(num_points, bins=bins, alpha=0.7, color='lightblue', 
             edgecolor='black', linewidth=0.5, density=True, label='Histogram')
    
    # Add kernel density estimation
    try:
        from scipy import stats
        density = stats.gaussian_kde(num_points)
        xs = np.linspace(num_points.min(), num_points.max(), 200)
        plt.plot(xs, density(xs), 'r-', linewidth=2, label='Kernel Density Estimation')
    except ImportError:
        print("Scipy not available, skipping KDE")
    
    plt.xlabel('Number of Points', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title('DrivaerML Dataset: Detailed Distribution of Number of Points', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add vertical lines for key statistics
    plt.axvline(num_points.mean(), color='red', linestyle='--', alpha=0.8, label='Mean')
    plt.axvline(num_points.median(), color='green', linestyle='--', alpha=0.8, label='Median')
    
    plt.tight_layout()
    
    # Save detailed histogram
    detailed_output_path = os.path.join(output_dir, 'num_points_detailed_histogram.png')
    plt.savefig(detailed_output_path, dpi=300, bbox_inches='tight')
    print(f"Detailed histogram saved to: {detailed_output_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return df

def main():
    parser = argparse.ArgumentParser(description="Plot distribution of number of points from drivaerml statistics")
    parser.add_argument('--csv_path', type=str, required=True,
                       help='Path to the CSV file containing statistics')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Directory to save the plots (default: same as CSV directory)')
    parser.add_argument('--no_show', action='store_true',
                       help='Do not display the plots')
    
    args = parser.parse_args()
    
    df = plot_num_points_distribution(
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        show_plot=not args.no_show
    )
    
    if df is not None:
        print("\nDataset Summary:")
        print(f"Total files: {len(df)}")
        print(f"Number of points - Min: {df['num_points'].min()}, Max: {df['num_points'].max()}")
        print(f"Mean: {df['num_points'].mean():.1f}, Median: {df['num_points'].median():.1f}")
        print(f"Standard deviation: {df['num_points'].std():.1f}")

if __name__ == '__main__':
    # Default configuration if run without arguments
    if len(os.sys.argv) == 1:
        # Default CSV path
        default_csv = "drivaerml_statistics.csv"
        
        if os.path.exists(default_csv):
            print(f"Using default CSV file: {default_csv}")
            plot_num_points_distribution(default_csv)
        else:
            print(f"Default CSV file '{default_csv}' not found.")
            print("Please run the statistics script first or provide CSV path as argument:")
            print("python plot_statistics.py --csv_path your_statistics.csv")
    else:
        main() 