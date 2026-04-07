export default async function handler(req, res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  if (req.method === 'OPTIONS') return res.status(200).end();

  const { tickers, period = '5y' } = req.query;
  if (!tickers) return res.status(400).json({ error: 'tickers required' });

  const tickerList = tickers.split(',').map(t => t.trim()).filter(Boolean).slice(0, 25);
  const results = {};

  const fetchOne = async (ticker) => {
    const urls = [
      `https://query1.finance.yahoo.com/v8/finance/chart/${encodeURIComponent(ticker)}?range=${period}&interval=1d`,
      `https://query2.finance.yahoo.com/v8/finance/chart/${encodeURIComponent(ticker)}?range=${period}&interval=1d`,
    ];
    for (const url of urls) {
      try {
        const r = await fetch(url, {
          headers: {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'application/json',
            'Accept-Language': 'en-US,en;q=0.9',
          },
          signal: AbortSignal.timeout(9000),
        });
        if (!r.ok) continue;
        const data = await r.json();
        const result = data?.chart?.result?.[0];
        if (!result) continue;
        const timestamps = result.timestamp;
        const adjClose = result.indicators?.adjclose?.[0]?.adjclose;
        const close = result.indicators?.quote?.[0]?.close;
        const prices = (adjClose || close)?.map(p => p ?? null);
        if (!timestamps || !prices || timestamps.length < 50) continue;
        return { timestamps, prices };
      } catch (_) { continue; }
    }
    return { error: `Failed: ${ticker}` };
  };

  // Batch to avoid rate limiting
  for (let i = 0; i < tickerList.length; i += 4) {
    const batch = tickerList.slice(i, i + 4);
    const out = await Promise.all(batch.map(fetchOne));
    batch.forEach((t, j) => { results[t] = out[j]; });
    if (i + 4 < tickerList.length) await new Promise(r => setTimeout(r, 300));
  }

  res.setHeader('Cache-Control', 's-maxage=3600, stale-while-revalidate=7200');
  res.json(results);
}
