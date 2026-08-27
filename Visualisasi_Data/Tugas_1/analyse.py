import plotly.express as px
import pandas as pd


df = pd.read_csv("huggingface_models.csv")

fig1 = px.scatter(df, x="param_count_b", y="downloads_30d", color="pipeline_tag", hover_data=["model_name"], title = "distribusi berdasarkan parameter count dan jumlah download 30 hari terakhir")
fig1.update_xaxes(range=[0, 1000])
fig1.update_yaxes(range=[0, 30_000_000])

fig2 = px.histogram(df, x="param_count_b", color="pipeline_tag", title = "distribusi berdasarkan parameter count")
fig2.update_xaxes(range=[0, 1000])

model_counts = (
    df.groupby(["created_year", "pipeline_tag"])
      .size()
      .reset_index(name="model_count")
)

fig3 = px.line(model_counts, x="created_year", y="model_count", color="pipeline_tag", title = "distribusi model berdasarkan tahun dibuat")
fig3.show()
