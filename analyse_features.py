import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns
from scipy.special import rel_entr

TEXTURE_FEATURES_3D = [
    # GLCM features
    "3GLCM_ACOR", "3GLCM_ASM", "3GLCM_CLUPROM", "3GLCM_CLUSHADE", "3GLCM_CLUTEND",
    "3GLCM_CONTRAST", "3GLCM_CORRELATION", "3GLCM_DIFAVE", "3GLCM_DIFENTRO", "3GLCM_DIFVAR",
    "3GLCM_DIS", "3GLCM_ENERGY", "3GLCM_ENTROPY", "3GLCM_HOM1", "3GLCM_HOM2",
    "3GLCM_ID", "3GLCM_IDN", "3GLCM_IDM", "3GLCM_IDMN", "3GLCM_INFOMEAS1", "3GLCM_INFOMEAS2",
    "3GLCM_IV", "3GLCM_JAVE", "3GLCM_JE", "3GLCM_JMAX", "3GLCM_JVAR",
    "3GLCM_SUMAVERAGE", "3GLCM_SUMENTROPY", "3GLCM_SUMVARIANCE", "3GLCM_VARIANCE",
    "3GLCM_ASM_AVE", "3GLCM_ACOR_AVE", "3GLCM_CLUPROM_AVE", "3GLCM_CLUSHADE_AVE",
    "3GLCM_CLUTEND_AVE", "3GLCM_CONTRAST_AVE", "3GLCM_CORRELATION_AVE",
    "3GLCM_DIFAVE_AVE", "3GLCM_DIFENTRO_AVE", "3GLCM_DIFVAR_AVE", "3GLCM_DIS_AVE",
    "3GLCM_ENERGY_AVE", "3GLCM_ENTROPY_AVE", "3GLCM_HOM1_AVE", "3GLCM_ID_AVE",
    "3GLCM_IDN_AVE", "3GLCM_IDM_AVE", "3GLCM_IDMN_AVE", "3GLCM_IV_AVE",
    "3GLCM_JAVE_AVE", "3GLCM_JE_AVE", "3GLCM_INFOMEAS1_AVE", "3GLCM_INFOMEAS2_AVE",
    "3GLCM_VARIANCE_AVE", "3GLCM_JMAX_AVE", "3GLCM_JVAR_AVE",
    "3GLCM_SUMAVERAGE_AVE", "3GLCM_SUMENTROPY_AVE", "3GLCM_SUMVARIANCE_AVE",

    # GLRLM features
    "3GLRLM_SRE", "3GLRLM_LRE", "3GLRLM_GLN", "3GLRLM_GLNN", "3GLRLM_RLN", "3GLRLM_RLNN",
    "3GLRLM_RP", "3GLRLM_GLV", "3GLRLM_RV", "3GLRLM_RE", "3GLRLM_LGLRE", "3GLRLM_HGLRE",
    "3GLRLM_SRLGLE", "3GLRLM_SRHGLE", "3GLRLM_LRLGLE", "3GLRLM_LRHGLE",
    "3GLRLM_SRE_AVE", "3GLRLM_LRE_AVE", "3GLRLM_GLN_AVE", "3GLRLM_GLNN_AVE",
    "3GLRLM_RLN_AVE", "3GLRLM_RLNN_AVE", "3GLRLM_RP_AVE", "3GLRLM_GLV_AVE",
    "3GLRLM_RV_AVE", "3GLRLM_RE_AVE", "3GLRLM_LGLRE_AVE", "3GLRLM_HGLRE_AVE",
    "3GLRLM_SRLGLE_AVE", "3GLRLM_SRHGLE_AVE", "3GLRLM_LRLGLE_AVE", "3GLRLM_LRHGLE_AVE",

    # GLSZM features
    "GLSZM_SAE", "GLSZM_LAE", "GLSZM_GLN", "GLSZM_GLNN", "GLSZM_SZN", "GLSZM_SZNN",
    "GLSZM_ZP", "GLSZM_GLV", "GLSZM_ZV", "GLSZM_ZE", "GLSZM_LGLZE", "GLSZM_HGLZE",
    "GLSZM_SALGLE", "GLSZM_SAHGLE", "GLSZM_LALGLE", "GLSZM_LAHGLE",

    # GLDM features
    "3GLDM_SDE", "3GLDM_LDE", "3GLDM_GLN", "3GLDM_DN", "3GLDM_DNN", "3GLDM_GLV",
    "3GLDM_DV", "3GLDM_DE", "3GLDM_LGLE", "3GLDM_HGLE", "3GLDM_SDLGLE", "3GLDM_SDHGLE",
    "3GLDM_LDLGLE", "3GLDM_LDHGLE",

    # GLDZM features
    "3GLDZM_SDE", "3GLDZM_LDE", "3GLDZM_LGLZE", "3GLDZM_HGLZE", "3GLDZM_SDLGLE",
    "3GLDZM_SDHGLE", "3GLDZM_LDLGLE", "3GLDZM_LDHGLE", "3GLDZM_GLNU", "3GLDZM_GLNUN",
    "3GLDZM_ZDNU", "3GLDZM_ZDNUN", "3GLDZM_ZP", "3GLDZM_GLM", "3GLDZM_GLV",
    "3GLDZM_ZDM", "3GLDZM_ZDV", "3GLDZM_ZDE",

    # NGLDM features
    "3NGLDM_LDE", "3NGLDM_HDE", "3NGLDM_LGLCE", "3NGLDM_HGLCE", "3NGLDM_LDLGLE",
    "3NGLDM_LDHGLE", "3NGLDM_HDLGLE", "3NGLDM_HDHGLE", "3NGLDM_GLNU", "3NGLDM_GLNUN",
    "3NGLDM_DCNU", "3NGLDM_DCNUN", "3NGLDM_DCP", "3NGLDM_GLM", "3NGLDM_GLV",
    "3NGLDM_DCM", "3NGLDM_DCV", "3NGLDM_DCENT", "3NGLDM_DCENE",

    # NGTDM features
    "3NGTDM_COARSENESS", "3NGTDM_CONTRAST", "3NGTDM_BUSYNESS", "3NGTDM_COMPLEXITY",
    "3NGTDM_STRENGTH",
]


original_features_dir = "../datasets/Craig_scans/features" 
normalized_features_dir = "../datasets/Craig_scans/normalized_features"

def read_features(features_dir: str) -> pd.DataFrame:
    """Read features from a directory.
    
    Args:
        features_dir: Directory containing features
        
    Returns:
        DataFrame containing all features
    """
    features = []
    for file in os.listdir(features_dir):
        df = pd.read_csv(os.path.join(features_dir, file))
        features.append(df)
    return pd.concat(features, ignore_index=True)


original_features = read_features(original_features_dir)
normalized_features = read_features(normalized_features_dir)

print(original_features.head())


def select_features(features: pd.DataFrame, feature_list: list) -> pd.DataFrame:
    """Select features from a DataFrame.
    
    Args:
        features: DataFrame containing features
        feature_list: List of feature names to select
        
    Returns:
        DataFrame containing selected features
    """ 
    return features[feature_list]


original_features = select_features(original_features, TEXTURE_FEATURES_3D)
normalized_features = select_features(normalized_features, TEXTURE_FEATURES_3D)

print(original_features.head())
print(normalized_features.head())



def plot_feature_histograms(original_features: pd.DataFrame, 
                          normalized_features: pd.DataFrame,
                          output_dir: str,
                          feature: str) -> None:
    """Plot histograms of features.
    
    Args:
        original_features: DataFrame with original features
        normalized_features: DataFrame with normalized features
        output_dir: Directory to save plots 
        n_features: Number of features to plot
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for feature in original_features.columns:
        plt.figure(figsize=(10, 5))
        plt.hist(original_features[feature], bins=30, alpha=0.5, label='Original')
        plt.hist(normalized_features[feature], bins=30, alpha=0.5, label='Normalized')
        plt.title(f'Histogram of {feature}')
        plt.legend()
        plt.savefig(os.path.join(output_dir, f'{feature}_histogram.png'))
        plt.close()


# def perform_ks_test(original_features: pd.DataFrame, 
#                    normalized_features: pd.DataFrame,
#                    output_dir: str) -> pd.DataFrame:
#     """Perform Kolmogorov-Smirnov test on features.
    
#     Args:
#         original_features: DataFrame with original features
#         normalized_features: DataFrame with normalized features
#         output_dir: Directory to save results
        
#     Returns:
#         DataFrame containing KS test results
#     """
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
        
#     results = []
#     for feature in original_features.columns:
#         # Perform KS test
#         statistic, pvalue = stats.ks_2samp(original_features[feature], 
#                                          normalized_features[feature])
        
#         # Calculate effect size (normalized statistic)
#         n1 = len(original_features[feature])
#         n2 = len(normalized_features[feature])
#         effect_size = statistic * np.sqrt((n1 * n2) / (n1 + n2))
        
#         # Calculate percentage change in mean and std
#         mean_change = ((normalized_features[feature].mean() - original_features[feature].mean()) 
#                       / original_features[feature].mean() * 100)
#         std_change = ((normalized_features[feature].std() - original_features[feature].std()) 
#                      / original_features[feature].std() * 100)
        
#         results.append({
#             'Feature': feature,
#             'KS_Statistic': statistic,
#             'P_Value': pvalue,
#             'Effect_Size': effect_size,
#             'Mean_Change_Percent': mean_change,
#             'Std_Change_Percent': std_change,
#             'Significant': pvalue < 0.05
#         })
    
#     # Create DataFrame and sort by effect size
#     results_df = pd.DataFrame(results)
#     results_df = results_df.sort_values('Effect_Size', ascending=False)
    
#     # Save results
#     results_df.to_csv(os.path.join(output_dir, 'ks_test_results.csv'), index=False)
    
#     # Create summary of significant changes
#     significant_changes = results_df[results_df['Significant']].copy()
#     significant_changes['Change_Level'] = pd.cut(
#         significant_changes['Effect_Size'],
#         bins=[-np.inf, 0.1, 0.3, 0.5, np.inf],
#         labels=['Small', 'Medium', 'Large', 'Very Large']
#     )
    
#     summary = significant_changes.groupby('Change_Level').size().to_frame('Count')
#     summary.to_csv(os.path.join(output_dir, 'ks_test_summary.csv'))
    
#     return results_df


# def analyze_feature_distribution(original_features: pd.DataFrame, 
#                                normalized_features: pd.DataFrame,
#                                feature_name: str,
#                                output_dir: str) -> dict:
#     """Analyze distribution of a single feature across original and normalized images.
    
#     Args:
#         original_features: DataFrame with original features
#         normalized_features: DataFrame with normalized features
#         feature_name: Name of the feature to analyze
#         output_dir: Directory to save plots
        
#     Returns:
#         Dictionary containing statistical test results
#     """
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
        
#     # Get values for the feature
#     orig_values = original_features[feature_name]
#     norm_values = normalized_features[feature_name]
    
#     # Create figure with two subplots
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
#     # Plot original values
#     ax1.hist(orig_values, bins='auto', alpha=0.7, color='blue', density=True)
#     ax1.plot(orig_values, stats.gaussian_kde(orig_values)(orig_values), 'r-', lw=2)
#     ax1.set_title(f'Original {feature_name} Distribution')
#     ax1.set_xlabel('Value')
#     ax1.set_ylabel('Density')
    
#     # Plot normalized values
#     ax2.hist(norm_values, bins='auto', alpha=0.7, color='green', density=True)
#     ax2.plot(norm_values, stats.gaussian_kde(norm_values)(norm_values), 'r-', lw=2)
#     ax2.set_title(f'Normalized {feature_name} Distribution')
#     ax2.set_xlabel('Value')
#     ax2.set_ylabel('Density')
    
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_dir, f'{feature_name}_distribution.png'))
#     plt.close()
    
#     # Calculate basic statistics
#     stats_dict = {
#         'original_mean': orig_values.mean(),
#         'original_std': orig_values.std(),
#         'original_cv': orig_values.std() / orig_values.mean(),
#         'normalized_mean': norm_values.mean(),
#         'normalized_std': norm_values.std(),
#         'normalized_cv': norm_values.std() / norm_values.mean(),
#     }
    
#     # Perform statistical tests
#     # 1. Shapiro-Wilk test for normality
#     _, orig_shapiro_p = stats.shapiro(orig_values)
#     _, norm_shapiro_p = stats.shapiro(norm_values)
    
#     # 2. Levene test for equal variances
#     _, levene_p = stats.levene(orig_values, norm_values)
    
#     # 3. T-test or Mann-Whitney U test based on normality
#     if orig_shapiro_p > 0.05 and norm_shapiro_p > 0.05:
#         # If both are normal, use t-test
#         _, ttest_p = stats.ttest_ind(orig_values, norm_values)
#         test_used = 'ttest'
#         test_p = ttest_p
#     else:
#         # If not normal, use Mann-Whitney U test
#         _, mw_p = stats.mannwhitneyu(orig_values, norm_values, alternative='two-sided')
#         test_used = 'mannwhitney'
#         test_p = mw_p
    
#     stats_dict.update({
#         'original_normality_p': orig_shapiro_p,
#         'normalized_normality_p': norm_shapiro_p,
#         'equal_variance_p': levene_p,
#         'distribution_test': test_used,
#         'distribution_test_p': test_p,
#         'same_distribution': test_p > 0.05
#     })
    
#     # Save statistics to file
#     pd.DataFrame([stats_dict]).to_csv(
#         os.path.join(output_dir, f'{feature_name}_stats.csv'), 
#         index=False
#     )
    
#     return stats_dict

# def compare_all_feature_distributions(original_features: pd.DataFrame,
#                                     normalized_features: pd.DataFrame,
#                                     output_dir: str) -> pd.DataFrame:
#     """Compare distributions of all features.
    
#     Args:
#         original_features: DataFrame with original features
#         normalized_features: DataFrame with normalized features
#         output_dir: Directory to save results
        
#     Returns:
#         DataFrame with comparison results
#     """
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
    
#     results = []
#     for feature in original_features.columns:
#         print(f"Analyzing feature: {feature}")
#         stats = analyze_feature_distribution(
#             original_features, 
#             normalized_features,
#             feature,
#             os.path.join(output_dir, 'distributions')
#         )
#         stats['feature'] = feature
#         results.append(stats)
    
#     # Create summary DataFrame
#     results_df = pd.DataFrame(results)
    
#     # Save summary
#     results_df.to_csv(os.path.join(output_dir, 'distribution_comparison.csv'), index=False)
    
#     # Create summary plot
#     plt.figure(figsize=(12, 6))
#     plt.bar(range(len(results_df)), 
#             results_df['distribution_test_p'],
#             alpha=0.6)
#     plt.axhline(y=0.05, color='r', linestyle='--', label='Significance Level (0.05)')
#     plt.xticks(range(len(results_df)), 
#                results_df['feature'], 
#                rotation=90)
#     plt.ylabel('P-value')
#     plt.title('Distribution Test P-values for All Features')
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(os.path.join(output_dir, 'distribution_pvalues.png'))
#     plt.close()
    
#     return results_df

# if __name__ == "__main__":
#     # Create output directory for analysis results
#     analysis_output_dir = "../datasets/Craig_scans/feature_analysis"
#     os.makedirs(analysis_output_dir, exist_ok=True)
    
#     # Read and select features
#     print("Reading features...")
#     original_features = read_features(original_features_dir)
#     normalized_features = read_features(normalized_features_dir)
    
#     original_features = select_features(original_features, TEXTURE_FEATURES_3D)
#     normalized_features = select_features(normalized_features, TEXTURE_FEATURES_3D)
    
#     # Compare distributions
#     print("\nComparing feature distributions...")
#     results = compare_all_feature_distributions(
#         original_features,
#         normalized_features,
#         analysis_output_dir
#     )
    
#     # Print summary
#     print("\nSummary of distribution comparisons:")
#     print(f"Total features analyzed: {len(results)}")
#     print(f"Features with significantly different distributions: {sum(~results['same_distribution'])}")
    
#     # Print details for features with different distributions
#     different_dist = results[~results['same_distribution']].sort_values('distribution_test_p')
#     if len(different_dist) > 0:
#         print("\nFeatures with significantly different distributions:")
#         print(different_dist[['feature', 'distribution_test', 'distribution_test_p']].head())

def calculate_kl_divergence(original_features: pd.DataFrame,
                          normalized_features: pd.DataFrame,
                          feature_name: str,
                          n_bins: int = 30,
                          output_dir: str = None) -> dict:
    """Calculate KL divergence between original and normalized feature distributions.
    
    Args:
        original_features: DataFrame with original features
        normalized_features: DataFrame with normalized features
        feature_name: Name of the feature to analyze
        n_bins: Number of bins for histogram
        output_dir: Optional directory to save visualization
        
    Returns:
        Dictionary containing KL divergence results and histogram data
    """
    # Get feature values
    orig_values = original_features[feature_name].values
    norm_values = normalized_features[feature_name].values
    
    # Calculate range for consistent binning
    min_val = min(orig_values.min(), norm_values.min())
    max_val = max(orig_values.max(), norm_values.max())
    bins = np.linspace(min_val, max_val, n_bins+1)
    
    # Calculate histograms
    orig_hist, _ = np.histogram(orig_values, bins=bins, density=True)
    norm_hist, _ = np.histogram(norm_values, bins=bins, density=True)
    
    # Add small constant to avoid zero probabilities
    epsilon = 1e-10
    orig_hist = orig_hist + epsilon
    norm_hist = norm_hist + epsilon
    
    # Normalize to get proper probability distributions
    orig_hist = orig_hist / orig_hist.sum()
    norm_hist = norm_hist / norm_hist.sum()
    
    # Calculate KL divergence in both directions
    kl_orig_to_norm = sum(rel_entr(orig_hist, norm_hist))
    kl_norm_to_orig = sum(rel_entr(norm_hist, orig_hist))
    
    # Calculate Jensen-Shannon divergence (symmetric)
    m_hist = 0.5 * (orig_hist + norm_hist)
    js_divergence = 0.5 * sum(rel_entr(orig_hist, m_hist)) + 0.5 * sum(rel_entr(norm_hist, m_hist))
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Plot histograms and KL divergence
        plt.figure(figsize=(12, 6))
        
        # Plot histograms
        bin_centers = (bins[:-1] + bins[1:]) / 2
        plt.bar(bin_centers, orig_hist, alpha=0.5, width=np.diff(bins)[0], 
               label='Original', color='blue')
        plt.bar(bin_centers, norm_hist, alpha=0.5, width=np.diff(bins)[0], 
               label='Normalized', color='green')
        
        plt.title(f'Feature Distribution Comparison: {feature_name}\n' + 
                 f'KL(P||Q)={kl_orig_to_norm:.4f}, KL(Q||P)={kl_norm_to_orig:.4f}\n' +
                 f'JS={js_divergence:.4f}')
        plt.xlabel('Value')
        plt.ylabel('Probability')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(os.path.join(output_dir, f'{feature_name}_kl_divergence.png'))
        plt.close()
    
    return {
        'feature': feature_name,
        'kl_orig_to_norm': kl_orig_to_norm,
        'kl_norm_to_orig': kl_norm_to_orig,
        'js_divergence': js_divergence,
        'hist_bins': bins,
        'orig_hist': orig_hist,
        'norm_hist': norm_hist
    }

def compare_feature_distributions_kl(original_features: pd.DataFrame,
                                   normalized_features: pd.DataFrame,
                                   output_dir: str) -> pd.DataFrame:
    """Compare distributions of all features using KL divergence.
    
    Args:
        original_features: DataFrame with original features
        normalized_features: DataFrame with normalized features
        output_dir: Directory to save results
        
    Returns:
        DataFrame with KL divergence results for all features
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    results = []
    for feature in original_features.columns:
        print(f"Analyzing KL divergence for: {feature}")
        kl_stats = calculate_kl_divergence(
            original_features,
            normalized_features,
            feature,
            output_dir=os.path.join(output_dir, 'kl_divergence')
        )
        results.append(kl_stats)
    
    # Create summary DataFrame
    results_df = pd.DataFrame(results)
    
    # Save detailed results
    results_df.to_csv(os.path.join(output_dir, 'kl_divergence_results.csv'), index=False)
    
    # Create summary plot of JS divergences
    plt.figure(figsize=(12, 6))
    plt.bar(range(len(results_df)), results_df['js_divergence'], alpha=0.6)
    plt.xticks(range(len(results_df)), results_df['feature'], rotation=90)
    plt.ylabel('Jensen-Shannon Divergence')
    plt.title('Distribution Differences Across Features')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'js_divergence_summary.png'))
    plt.close()
    
    return results_df

if __name__ == "__main__":
    # Create output directory for analysis results
    analysis_output_dir = "../datasets/Craig_scans/feature_analysis"
    os.makedirs(analysis_output_dir, exist_ok=True)
    
    # Read and select features
    print("Reading features...")
    original_features = read_features(original_features_dir)
    normalized_features = read_features(normalized_features_dir)
    
    original_features = select_features(original_features, TEXTURE_FEATURES_3D)
    normalized_features = select_features(normalized_features, TEXTURE_FEATURES_3D)
    
    # Compare distributions using KL divergence
    print("\nComparing feature distributions using KL divergence...")
    results = compare_feature_distributions_kl(
        original_features,
        normalized_features,
        analysis_output_dir
    )
    
    # Print summary
    print("\nTop 5 features with largest distribution changes (by JS divergence):")
    print(results.nlargest(5, 'js_divergence')[['feature', 'js_divergence', 'kl_orig_to_norm', 'kl_norm_to_orig']])