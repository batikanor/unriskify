FROM node:20-alpine

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci

COPY . .

# Set environment variables for Vite to listen on all interfaces
ENV VITE_HOST=0.0.0.0
ENV VITE_PORT=5173

# Use Vite's development server instead of building and serving
CMD ["npm", "run", "dev"]