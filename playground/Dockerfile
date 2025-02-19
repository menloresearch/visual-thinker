# Use official Node.js image
FROM node:18-alpine

# Set working directory
WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy all files
COPY . .

# Build the application
RUN npm run build

# Expose the port
EXPOSE 3000

# Set default API URL
ENV API_URL=https://visual-thinker-be.menlo.ai

# Start the application
CMD ["npm", "start"]
