# thông số mô hình cơ bản
# import matplotlib.pyplot as plt

# # Dữ liệu
# splits = ['split 0', 'split 1', 'split 2', 'split 3', 'split 4']
# tvsum_f = [0.5788, 0.5483, 0.6389, 0.5967, 0.5958]
# tvsum_d = [0.5074, 0.4702, 0.4787, 0.4537, 0.4712]

# summe_f = [0.4898, 0.4901, 0.5971, 0.6333, 0.4366]
# summe_d = [0.6173, 0.6115, 0.7116, 0.5955, 0.6791]

# # Vẽ biểu đồ
# plt.figure(figsize=(12, 6))

# # TVSum
# plt.subplot(1, 2, 1)
# plt.plot(splits, tvsum_f, marker='o', label='F-score')
# plt.plot(splits, tvsum_d, marker='s', label='Diversity')
# plt.title('TVSum')
# plt.ylim(0.4, 0.7)
# plt.xlabel('Split')
# plt.ylabel('Score')
# plt.legend()
# plt.grid(True)

# # SumMe
# plt.subplot(1, 2, 2)
# plt.plot(splits, summe_f, marker='o', label='F-score')
# plt.plot(splits, summe_d, marker='s', label='Diversity')
# plt.title('SumMe')
# plt.ylim(0.4, 0.75)
# plt.xlabel('Split')
# plt.ylabel('Score')
# plt.legend()
# plt.grid(True)

# plt.tight_layout()
# plt.show()
################################################################################################
#thông số mô hình custom
# import matplotlib.pyplot as plt

# # Dữ liệu đầu vào
# splits = [0, 1, 2, 3, 4]
# diversity = [0.5712, 0.7782, 0.6432, 0.7782, 0.7782]
# f_score = [1.0000, 0.9375, 0.5435, 0.9375, 0.9375]

# # Trung bình tổng hợp
# avg_diversity = 0.7098
# avg_f_score = 0.8712

# # Vẽ đồ thị
# plt.figure(figsize=(10, 5))
# plt.plot(splits, diversity, marker='o', label='Diversity')
# plt.plot(splits, f_score, marker='s', label='F-score')
# plt.hlines(avg_diversity, xmin=0, xmax=4, colors='blue', linestyles='dashed', label='Avg Diversity')
# plt.hlines(avg_f_score, xmin=0, xmax=4, colors='orange', linestyles='dashed', label='Avg F-score')

# plt.xticks(splits)
# plt.xlabel("Split Index")
# plt.ylabel("Giá trị")
# plt.title("Biểu đồ Diversity và F-score theo các split")
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()
################################################################################################
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# data = [
#     ["Self-Attention", 1.0, 1.0, 0.6250],
#     ["Self-Attention", 1.0, 0.0, 0.6179],
#     ["Self-Attention", 0.0, 1.0, 0.6048],
#     ["Self-Attention", 0.0, 0.0, 0.6057],
#     ["Bi-LSTM", 1.0, 1.0, 0.6150],
#     ["Bi-LSTM", 1.0, 0.0, 0.6062],
#     ["Bi-LSTM", 0.0, 1.0, 0.5978],
#     ["Bi-LSTM", 0.0, 0.0, 0.5985],
#     ["GCN", 1.0, 1.0, 0.6188],
#     ["GCN", 1.0, 0.0, 0.6128],
#     ["GCN", 0.0, 1.0, 0.5967],
#     ["GCN", 0.0, 0.0, 0.5991],
# ]

# df = pd.DataFrame(data, columns=["Mô hình", "λ", "μ", "F-score"])
# df["Cấu hình"] = df["λ"].astype(str) + "-" + df["μ"].astype(str)

# # Vẽ biểu đồ
# plt.figure(figsize=(12, 6))
# ax = sns.barplot(
#     data=df,
#     x="Mô hình",
#     y="F-score",
#     hue="Cấu hình",
#     palette="Set2",
#     edgecolor="gray"
# )

# # Thêm giá trị số trên mỗi cột
# for p in ax.patches:
#     height = p.get_height()
#     ax.annotate(f"{height:.3f}",
#                 (p.get_x() + p.get_width() / 2., height),
#                 ha='center', va='bottom', fontsize=9, color='black', xytext=(0, 2),
#                 textcoords='offset points')

# plt.title("F-score của các mô hình Anchor-Free theo λ và μ (TVSum)", fontsize=14, fontweight='bold')
# plt.ylabel("F-score", fontsize=12)
# plt.xlabel("Mô hình", fontsize=12)
# plt.ylim(0.59, 0.63)
# plt.grid(axis='y', linestyle='--', alpha=0.4)
# plt.legend(title="Cấu hình λ-μ")
# plt.tight_layout()
# plt.show()

###################################################################################
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df_summe = pd.DataFrame([
    ["Self-Attention", 1.0, 1.0, 0.5850],
    ["Self-Attention", 1.0, 0.0, 0.5782],
    ["Self-Attention", 0.0, 1.0, 0.5665],
    ["Self-Attention", 0.0, 0.0, 0.5693],
    ["Bi-LSTM", 1.0, 1.0, 0.5741],
    ["Bi-LSTM", 1.0, 0.0, 0.5626],
    ["Bi-LSTM", 0.0, 1.0, 0.5555],
    ["Bi-LSTM", 0.0, 0.0, 0.5539],
    ["GCN", 1.0, 1.0, 0.5793],
    ["GCN", 1.0, 0.0, 0.5731],
    ["GCN", 0.0, 1.0, 0.5522],
    ["GCN", 0.0, 0.0, 0.5547],
], columns=["Mô hình", "λ", "μ", "F-score"])

# Tạo cột biểu diễn cấu hình lambda - mu
df_summe["Cấu hình"] = df_summe["λ"].astype(str) + "-" + df_summe["μ"].astype(str)

# Vẽ biểu đồ
plt.figure(figsize=(12, 6))
ax = sns.barplot(
    data=df_summe,
    x="Mô hình",
    y="F-score",
    hue="Cấu hình",
    palette="Set2",
    edgecolor="gray"
)

# Ghi giá trị trên đầu cột
for p in ax.patches:
    height = p.get_height()
    ax.annotate(f"{height:.3f}",
                (p.get_x() + p.get_width() / 2., height),
                ha='center', va='bottom',
                fontsize=9, color='black', xytext=(0, 2),
                textcoords='offset points')

# Thiết lập biểu đồ
plt.title("F-score của các mô hình Anchor-Free theo λ và μ (SumMe)", fontsize=14, fontweight='bold')
plt.xlabel("Mô hình", fontsize=12)
plt.ylabel("F-score", fontsize=12)
plt.ylim(0.54, 0.59)
plt.grid(axis='y', linestyle='--', alpha=0.4)
plt.legend(title="Cấu hình λ-μ")
plt.tight_layout()
plt.show()

