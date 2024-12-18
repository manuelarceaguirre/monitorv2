FROM public.ecr.aws/lambda/python:3.10

# Copy requirements first to leverage Docker cache
COPY requirements.txt ${LAMBDA_TASK_ROOT}

# Install packages with optimizations
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy function code
COPY app.py ${LAMBDA_TASK_ROOT}

# Set the CMD to your handler
CMD [ "app.lambda_handler" ]