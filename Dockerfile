FROM tensorflow/tensorflow:2.19.0-gpu-jupyter

# Cài các thư viện bạn cần
RUN pip install --upgrade pip && \
    pip install pandas scikit-learn

# Mặc định chạy JupyterLab
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser"]

EXPOSE 8888