# 使用 tiangolo/uwsgi-nginx-flask 镜像作为基础镜像
FROM tiangolo/uwsgi-nginx-flask:python3.9

# 设置工作目录为 /app
WORKDIR /app

# 将当前目录下的 app 目录复制到镜像的 /app 目录中
COPY ./app /app

# 安装应用的依赖
RUN pip install --no-cache-dir -r /app/tmp/requirements.txt

# 声明容器运行时监听的端口为 80
EXPOSE 80

# 定义容器启动时运行的命令为启动 app.py
CMD ["python", "app.py"]
