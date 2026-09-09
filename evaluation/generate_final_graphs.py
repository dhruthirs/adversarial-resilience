import matplotlib.pyplot as plt


# ============================================================
# FINAL CNN ATTACK RESULTS
# ============================================================

attacks = ["Clean", "FGSM", "PGD", "GAN"]
accuracy = [98.55, 12.91, 0.00, 71.64]

asr_attacks = ["FGSM", "PGD", "GAN"]
asr = [86.90, 100.00, 27.58]


# ============================================================
# ACCURACY GRAPH
# ============================================================

plt.figure(figsize=(8, 5))

bars = plt.bar(attacks, accuracy)

plt.ylabel("Accuracy (%)")
plt.xlabel("Attack")
plt.title("CNN Accuracy: Clean vs Adversarial Attacks")
plt.ylim(0, 105)

for bar, value in zip(bars, accuracy):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 1.5,
        f"{value:.2f}%",
        ha="center"
    )

plt.tight_layout()
plt.savefig(
    "evaluation/cnn_accuracy_comparison.png",
    dpi=300
)

plt.close()


# ============================================================
# ASR GRAPH
# ============================================================

plt.figure(figsize=(8, 5))

bars = plt.bar(asr_attacks, asr)

plt.ylabel("Attack Success Rate (%)")
plt.xlabel("Attack")
plt.title("CNN Attack Success Rate Comparison")
plt.ylim(0, 110)

for bar, value in zip(bars, asr):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        value + 2,
        f"{value:.2f}%",
        ha="center"
    )

plt.tight_layout()
plt.savefig(
    "evaluation/cnn_asr_comparison.png",
    dpi=300
)

plt.close()


print("========================================")
print("FINAL GRAPHS GENERATED SUCCESSFULLY")
print("========================================")
print("evaluation/cnn_accuracy_comparison.png")
print("evaluation/cnn_asr_comparison.png")