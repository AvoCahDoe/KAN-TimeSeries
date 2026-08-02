# TimeKAN Portfolio Showcase

Light results page for TimeKAN: intro → math → interactive RMSE charts → ablations / figures → tradeoffs.

## Run

```bash
cd KAN-TimeSeries/showcase
npm install
npm run dev
```

## Build

```bash
npm run build
npm run preview
```

## Configure

Edit `src/site.ts` for `githubUrl`. Metrics snapshot from `../results.md`; figures in `public/figures/`.

## Deploy

Vercel: root directory `KAN-TimeSeries/showcase`, build `npm run build`, output `dist`.
