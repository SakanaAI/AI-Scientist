# AI-Scientist Frontend

This is the frontend application for the AI-Scientist project, a comprehensive system for automated scientific discovery.

## Features

- **User Management**: Authentication and profile management
- **Dashboard**: Overview of research activities and progress
- **Experiment Management**: Create, run, and monitor scientific experiments
- **Paper Generation**: Create and edit scientific papers based on experiment results
- **Results Visualization**: Interactive visualization of experimental results

## Getting Started

### Prerequisites

- Node.js (v16 or later)
- npm (v8 or later)

### Installation

1. Navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm start
   ```
   Alternatively, use the provided script:
   ```bash
   ./run-frontend.sh
   ```

4. The application will be available at [http://localhost:3000](http://localhost:3000)

## Build for Production

To create a production build:

```bash
npm run build
```

The build files will be generated in the `dist` directory.

## Project Structure

```
frontend/
├── public/                   # Static files
├── src/                      # Source code
│   ├── components/           # UI components
│   │   ├── layout/           # Layout components (e.g., MainLayout, AuthLayout)
│   │   ├── dashboard/        # Dashboard-specific components
│   │   ├── experiments/      # Experiment-related components
│   │   ├── papers/           # Paper-related components
│   │   ├── auth/             # Authentication components
│   │   ├── common/           # Common/shared components
│   │   └── results/          # Result visualization components
│   ├── pages/                # Page components
│   ├── services/             # API services and data fetching
│   ├── context/              # React context providers
│   ├── hooks/                # Custom React hooks
│   ├── utils/                # Utility functions
│   └── assets/               # Static assets like images and styles
├── dist/                     # Production build output
└── package.json              # NPM dependencies and scripts
```

## Technologies Used

- **React**: Frontend library for building user interfaces
- **TypeScript**: Type-safe JavaScript
- **Material UI**: React UI framework for faster and easier web development
- **React Router**: Declarative routing for React
- **Axios**: Promise-based HTTP client for making API requests
- **Chart.js**: Charting library for data visualization

## Authentication

For demo purposes, the frontend uses a mock authentication service. In a production environment, this would be replaced with a real authentication API.

Demo credentials:
- Email: demo@example.com
- Password: password

## Connected Backend

The frontend is designed to communicate with the AI-Scientist backend. By default, it connects to `http://localhost:5000/api`. You can configure the API endpoint by setting the `REACT_APP_API_URL` environment variable.

## Contributing

Contributions to the AI-Scientist frontend are welcome. Please follow these steps:

1. Fork the repository
2. Create a new branch
3. Implement your changes
4. Submit a pull request

## License

This project is licensed under the terms specified in the LICENSE file at the root of the repository. 