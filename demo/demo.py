#!/usr/bin/env python3
"""
NumCompute Demo: Complete ML Workflow

This script demonstrates the full NumCompute workflow:
1. Load CSV data with missing values  
2. Preprocessing pipeline (impute → scale)
3. Model evaluation with classification metrics
4. Statistical analysis

Run with: python examples/demo.py
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

try:
    from numcompute import (
        load_csv, 
        SimpleImputer, 
        StandardScaler, 
        Pipeline, 
        accuracy, 
        precision, 
        recall, 
        f1, 
        confusion_matrix, 
        descriptive_stats
    )
    print("✅ NumCompute imported successfully!")
except ImportError:
    print("❌ NumCompute not found. Install with: pip install -e .")
    exit(1)


def create_sample_data():
    """Create realistic demo dataset with missing values."""
    csv_data = '''age,weight,income,class
25,,45000,A
30,70,,B
35,65,60000,A
,80,55000,B
28,75,,A'''
    
    data_file = Path("demo_data.csv")
    data_file.write_text(csv_data)
    return data_file


def main():
    print("🚀 NumCompute Demo - End-to-End ML Workflow")
    print("=" * 50)
    
    # 1. CREATE & LOAD DATA
    print("\n📥 1. Loading CSV data with missing values...")
    data_file = create_sample_data()
    
    X = load_csv(data_file)
    y = load_csv(data_file, usecols=[3]).flatten()
    
    print(f"   Features shape: {X.shape}")
    print(f"   Target shape:   {y.shape}")
    print(f"   Missing values: {np.isnan(X).sum()}")
    print(f"   Raw data preview:\n{X}")
    
    # 2. PREPROCESSING PIPELINE
    print("\n🔄 2. Building preprocessing pipeline...")
    pipe = Pipeline([
        ("impute", SimpleImputer(strategy="mean")),
        ("scale", StandardScaler()),
    ])
    
    X_clean = pipe.fit_transform(X)
    print(f"   Clean data shape: {X_clean.shape}")
    print(f"   No missing values: {np.isnan(X_clean).sum()}")
    print(f"   Clean data preview:\n{X_clean}")
    
    # 3. DESCRIPTIVE STATISTICS
    print("\n📊 3. Computing descriptive statistics...")
    stats = descriptive_stats(X_clean)
    print("   Statistics:")
    for key, value in stats.items():
        print(f"     {key:10}: {value}")
    
    # 4. MODEL EVALUATION (simulated predictions)
    print("\n🎯 4. Model evaluation...")
    np.random.seed(42)
    y_pred = np.random.choice(['A', 'B'], size=len(y))
    
    acc = accuracy(y, y_pred)
    prec = precision(y, y_pred)
    rec = recall(y, y_pred)
    f1_score = f1(y, y_pred)
    
    print(f"   Accuracy:  {acc:.1%}")
    print(f"   Precision: {prec:.3f}")
    print(f"   Recall:    {rec:.3f}")
    print(f"   F1:        {f1_score:.3f}")
    
    cm = confusion_matrix(y, y_pred)
    print(f"   Confusion Matrix:\n{cm}")
    
    # 5. VISUALIZATION
    print("\n📈 5. Creating visualizations...")
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Feature distributions
        feature_names = ['age', 'weight', 'income']
        for i in range(min(3, X_clean.shape[1])):
            axes[i//2, i%2].hist(X_clean[:, i], bins=8, alpha=0.7, color='skyblue')
            axes[i//2, i%2].set_title(f'{feature_names[i]} distribution')
            axes[i//2, i%2].set_xlabel('Scaled value')
            axes[i//2, i%2].set_ylabel('Frequency')
        
        # Class distribution
        class_counts = np.bincount([0 if c=='A' else 1 for c in y])
        axes[1, 1].pie(class_counts, labels=['Class A', 'Class B'], autopct='%1.1f%%')
        axes[1, 1].set_title('Class Distribution')
        
        plt.tight_layout()
        plt.savefig('demo_results.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("   📊 Plot saved as 'demo_results.png'")
        
    except Exception as e:
        print(f"   ⚠️  Plotting skipped: {e}")
    
    # 6. ADVANCED FEATURES
    print("\n⚡ 6. Advanced features demo...")
    from numcompute import rank, topk
    
    scores = np.array([90, 85, 85, 92, 78])
    print(f"   Scores: {scores}")
    print(f"   Average ranks: {rank(scores, 'average')}")
    print(f"   Dense ranks:    {rank(scores, 'dense')}")
    
    top_scores, top_idx = topk(scores, k=2)
    print(f"   Top 2 scores: {top_scores} (indices: {top_idx})")
    
    print("\n🎉 Demo complete! Check 'demo_results.png' for plots.")
    print("\n💡 All NumCompute functions used: load_csv, Pipeline, Imputer, Scaler, "
          "accuracy, precision, recall, f1, confusion_matrix, descriptive_stats, rank, topk")


if __name__ == "__main__":
    main()