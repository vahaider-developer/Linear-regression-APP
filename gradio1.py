import gradio as gr
import joblib
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

model_tesla = joblib.load('tesla_model.pkl')
model_adv = joblib.load('adv_model.pkl')

def predict_tesla(year, km):
    pred = model_tesla.predict([[year, km]])
    return f"Predicted Price: ${pred[0]:,.2f}"

def predict_sales(tv, radio, newspaper):
    pred = model_adv.predict([[tv, radio, newspaper]])
    return f"Predicted Sales: {pred[0]:.2f} Units"

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Dual Dataset Regression Predictor")
    
    with gr.Tab("Tesla Price Predictor"):
        with gr.Row():
            year = gr.Number(label="Manufacturing Year (e.g., 2021)")
            km = gr.Number(label="Kilometers Driven")
        btn_tesla = gr.Button("Predict Price")
        out_tesla = gr.Textbox(label="Prediction Result")
        btn_tesla.click(predict_tesla, inputs=[year, km], outputs=out_tesla)

    with gr.Tab("Advertising Sales Predictor"):
        with gr.Row():
            tv = gr.Number(label="TV Budget ($)")
            radio = gr.Number(label="Radio Budget ($)")
            newspaper = gr.Number(label="Newspaper Budget ($)")
        btn_sales = gr.Button("Predict Sales")
        out_sales = gr.Textbox(label="Prediction Result")
        btn_sales.click(predict_sales, inputs=[tv, radio, newspaper], outputs=out_sales)

if __name__ == "__main__":
    demo.launch()