# Use official Node.js image
FROM node:18-alpine

# Set working directory
WORKDIR /app

# Copy package files
COPY package.json ./

# Install dependencies
RUN npm ci

# Copy all files
COPY . .

# Build the application
RUN npm run build

# Expose the port
EXPOSE 3000

# Set default API URL
ENV API_URL=http://localhost:8000

# Start the application
CMD ["next", "start"]
