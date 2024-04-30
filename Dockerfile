# 使用 tiangolo/uwsgi-nginx-flask 镜像作为基础镜像
FROM tiangolo/uwsgi-nginx-flask:python3.8

# 将当前目录下的 app 目录复制到镜像的 /app 目录中
COPY ./app /app

# 更新 pip，并安装虚拟环境工具
RUN pip install --no-cache-dir --upgrade pip virtualenv

# 在容器中创建虚拟环境
RUN virtualenv /venv

# 激活虚拟环境
ENV PATH="/venv/bin:$PATH"

# 在虚拟环境中安装应用程序所需的 Python 依赖项
RUN pip install --no-cache-dir -r /app/requirements.txt
