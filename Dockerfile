# 使用 tiangolo/uwsgi-nginx-flask 镜像作为基础镜像
FROM tiangolo/uwsgi-nginx-flask:python3.8

# 将当前目录下的 app 目录复制到镜像的 /app 目录中
COPY ./app /app

# 安装应用程序所需的 Python 依赖项
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r /app/requirements.txt
