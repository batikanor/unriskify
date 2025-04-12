FROM node:20-alpine

WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci

COPY . .

# Use Vite's development server instead of building and serving
CMD ["npm", "run", "dev"]