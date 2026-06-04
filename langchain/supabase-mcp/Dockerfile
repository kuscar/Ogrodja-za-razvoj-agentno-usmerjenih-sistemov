FROM node:20-alpine

WORKDIR /app

RUN npm install -g @supabase/mcp-server-supabase supergateway@1

EXPOSE 8080

CMD ["supergateway", \
     "--stdio", "npx @supabase/mcp-server-supabase", \
     "--port", "8080"]
