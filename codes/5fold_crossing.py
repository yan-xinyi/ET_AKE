import re
import glob
import pandas as pd

# 匹配 PRF 行的正则
pattern = re.compile(r"Best@(\d+): P=([\d.]+), R=([\d.]+), F=([\d.]+)")
datapath = "./result/log/Abstract320/ATT_BiLSTM/train*-*.txt"
cvpath = "./result/log/Abstract320/ATT_BiLSTM/cv_prf_results.xlsx"
sigpath = "./result/log/Abstract320/ATT_BiLSTM/attbl_sig_results.xlsx"  # 改成对应文件夹名字

rows = []
for fname in glob.glob(datapath):  # 假设文件名是 train1-featureA.txt
    feature = fname.split("-")[1].split(".")[0]
    fold = int(re.search(r"train(\d+)", fname).group(1))

    with open(fname, "r", encoding="utf-8") as f:
        for line in f:
            m = pattern.search(line)
            if m:
                k, p, r, f1 = m.groups()
                rows.append({
                    "Feature": feature,
                    "Fold": fold,
                    "TopK": int(k),
                    "P": float(p),
                    "R": float(r),
                    "F": float(f1)
                })

df = pd.DataFrame(rows)

# 按 feature + topk 聚合，算均值和标准差
agg_df = df.groupby(["Feature", "TopK"]).agg(["mean", "std"]).reset_index()

# 格式化 mean ± std
def format_mean_std(mean, std, decimals=2):
    return f"{mean:.{decimals}f}±{std:.{decimals}f}"


pretty_df = pd.DataFrame({
    "Feature": agg_df["Feature"],
    "TopK": agg_df["TopK"],
    "P": [format_mean_std(m, s) for m, s in zip(agg_df["P"]["mean"], agg_df["P"]["std"])],
    "R": [format_mean_std(m, s) for m, s in zip(agg_df["R"]["mean"], agg_df["R"]["std"])],
    "F": [format_mean_std(m, s) for m, s in zip(agg_df["F"]["mean"], agg_df["F"]["std"])],
})

# 排序：先按 Feature，再按 TopK
pretty_df = pretty_df.sort_values(by=["Feature", "TopK"]).reset_index(drop=True)

print(pretty_df.to_string(index=False))

# 保存到 Excel
pretty_df.to_excel(cvpath, index=False)


import pandas as pd
from scipy.stats import ttest_rel, wilcoxon


def significance_test(df, metric="F", features=None, topk_list=None, test="ttest"):
    """
    基于五折结果进行显著性检验
    df: 包含 Feature, Fold, TopK, metric 列的 DataFrame
    metric: 指标（P/R/F）
    features: 需要比较的两个特征组合，例如 ["featureA", "featureB"]
    topk_list: 要比较的 TopK 列表，例如 [3, 5, 10]
    test: "ttest" 或 "wilcoxon"
    """
    if features is None or len(features) != 2:
        raise ValueError("需要传入要比较的两个 features，例如 ['featureA', 'featureB']")
    if topk_list is None:
        topk_list = sorted(df["TopK"].unique())

    results = []
    for k in topk_list:
        vals1 = df[(df["Feature"] == features[0]) & (df["TopK"] == k)][metric].values
        vals2 = df[(df["Feature"] == features[1]) & (df["TopK"] == k)][metric].values

        if len(vals1) != len(vals2):
            raise ValueError(f"TopK={k} 时两组折数不一致，无法配对比较")

        if test == "ttest":
            stat, p = ttest_rel(vals1, vals2)
        else:  # 默认 Wilcoxon
            stat, p = wilcoxon(vals1, vals2)

        results.append({
            "TopK": k,
            f"{features[0]}_{metric}_mean": vals1.mean(),
            f"{features[1]}_{metric}_mean": vals2.mean(),
            "p_value": p
        })

    return pd.DataFrame(results)

# 假设 df 是你收集的所有 row 数据
sig_df1 = significance_test(df, metric="F", features=["no feature", "FFD"], topk_list=[3, 5, 10], test="ttest")
sig_df2 = significance_test(df, metric="F", features=["no feature", "FN"], topk_list=[3, 5, 10], test="ttest")
sig_df3 = significance_test(df, metric="F", features=["no feature", "TFD"], topk_list=[3, 5, 10], test="ttest")
sig_df4 = significance_test(df, metric="F", features=["no feature", "FFD_FN"], topk_list=[3, 5, 10], test="ttest")
sig_df5 = significance_test(df, metric="F", features=["no feature", "FN_TFD"], topk_list=[3, 5, 10], test="ttest")
sig_df6 = significance_test(df, metric="F", features=["no feature", "FFD_TFD"], topk_list=[3, 5, 10], test="ttest")
sig_df7 = significance_test(df, metric="F", features=["no feature", "FFD_FN_TFD"], topk_list=[3, 5, 10], test="ttest")

sig_df1["feature"] = "FFD"
sig_df2["feature"] = "FN"
sig_df3["feature"] = "TFD"
sig_df4["feature"] = "FFD_FN"
sig_df5["feature"] = "FN_TFD"
sig_df6["feature"] = "FFD_TFD"
sig_df7["feature"] = "FFD_FN_TFD"
# "FN", "TFD", "FFD+FN", "FN+TFD", "FFD+TFD", "FFD+FN+TFD"
# 合并
all_sig = pd.concat([sig_df1, sig_df2, sig_df3, sig_df4, sig_df5, sig_df6, sig_df7], ignore_index=True)

with pd.ExcelWriter(sigpath) as writer:
     all_sig.to_excel(writer, sheet_name="all_features")