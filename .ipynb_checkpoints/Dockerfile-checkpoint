# Base image với GPU + JupyterLab
FROM tensorflow/tensorflow:2.19.0-gpu-jupyter

# Cập nhật pip + cài thêm pandas, scikit-learn
RUN pip install --upgrade pip && \
    pip install pandas scikit-learn

# Mở port 8888 cho JupyterLab
EXPOSE 8888