# Predli Research Agent - Frontend

## Quick Start

```bash
# Install dependencies
npm install

# Start development server
npm run dev
```

Frontend will be available at: http://localhost:3000

## Requirements

- Node.js 18+
- npm or yarn
- Backend API running at http://localhost:8000

## Features

- ✅ Landing page with hero and features
- ✅ Research interface with context selector
- ✅ Real-time token counter
- ✅ PDF download
- ✅ Feedback system
- ✅ Responsive design
- ✅ Smooth animations

## Structure

```
frontend/
├── app/
│   ├── layout.tsx          # Root layout with nav
│   ├── page.tsx            # Landing page
│   ├── research/
│   │   └── page.tsx        # Research interface
│   └── globals.css         # Global styles
├── components/
│   ├── ResearchForm.tsx    # Research form
│   ├── ResultsDisplay.tsx  # Results & feedback
│   └── TokenCounter.tsx    # Token display
└── package.json
```

## Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm start` - Start production server
- `npm run lint` - Run ESLint

## Configuration

No additional configuration needed. The frontend expects the backend API at `http://localhost:8000`.

To change the API URL, update the fetch calls in:

- `app/research/page.tsx`
- `components/ResultsDisplay.tsx`

## Color Scheme

- **Primary**: Black (#000000)
- **Secondary**: White (#FFFFFF)
- **Accent**: #cb4f2b
- **Background**: Zinc-900 (#18181b)
- **Borders**: Zinc-800 (#27272a)
