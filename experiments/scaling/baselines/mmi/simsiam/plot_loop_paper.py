from plot_results_paper import plotspu

# Paper fine-tuning configurations:
# ft=0: all layers frozen (test pre-trained features only)
# ft=5: all layers trainable (full fine-tuning)

# Test with all trainable first
for layers in [5]:
    plotspu(layers)
