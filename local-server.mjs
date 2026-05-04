import http from 'node:http';
import fs from 'node:fs/promises';
import handler from './api/prices.js';

const port = Number(process.env.PORT || 4173);

const server = http.createServer(async (req, res) => {
  try {
    const url = new URL(req.url, `http://127.0.0.1:${port}`);

    if (url.pathname === '/api/prices') {
      const query = Object.fromEntries(url.searchParams.entries());
      const vReq = { method: req.method, query };
      const vRes = {
        setHeader: (key, value) => res.setHeader(key, value),
        status(code) {
          res.statusCode = code;
          return this;
        },
        json(payload) {
          res.setHeader('Content-Type', 'application/json; charset=utf-8');
          res.end(JSON.stringify(payload));
        },
      };

      await handler(vReq, vRes);
      return;
    }

    if (url.pathname.startsWith('/assets/')) {
      const fileName = url.pathname.replace(/^\/assets\//, '');
      const asset = await fs.readFile(`assets/${fileName}`);
      const type = fileName.endsWith('.png') ? 'image/png' : 'application/octet-stream';
      res.setHeader('Content-Type', type);
      res.end(asset);
      return;
    }

    const html = await fs.readFile('index.html', 'utf8');
    res.setHeader('Content-Type', 'text/html; charset=utf-8');
    res.end(html);
  } catch (error) {
    res.statusCode = 500;
    res.end(String(error?.stack || error));
  }
});

server.listen(port, '127.0.0.1', () => {
  console.log(`Portfolio Investment System running at http://127.0.0.1:${port}`);
});
