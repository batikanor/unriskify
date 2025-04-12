FROM node:20-alpine

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm install
RUN npm install react-force-graph-3d three @types/three

# Copy app files
COPY . .

# Vite uses port 5173 by default
EXPOSE 5173

# Set environment variables
ENV HOST=0.0.0.0
ENV PORT=5173

# Start the development server
CMD ["npm", "run", "dev"]