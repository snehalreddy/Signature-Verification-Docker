# Use an official Python runtime as a parent image
FROM python:slim
# Set the working directory to /app
WORKDIR /app

COPY ./requirements.txt /app/requirements.txt

# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Copy the current directory contents into the container at /app
COPY ./flask_app /app/flask_app
# Make port 8000 available to the world outside this container
EXPOSE 5000
# Run app.py when the container launches
CMD ["python", "./flask_app/app.py"]