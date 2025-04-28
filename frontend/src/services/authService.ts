import axios from 'axios';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000/api';

interface User {
  id: string;
  username: string;
  email: string;
}

interface LoginCredentials {
  email: string;
  password: string;
}

interface RegisterCredentials {
  username: string;
  email: string;
  password: string;
}

// Configure axios to include credentials
axios.defaults.withCredentials = true;

// Add a request interceptor to include the auth token
axios.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('token');
    if (token) {
      config.headers['Authorization'] = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// For demonstration purposes, this is a mock implementation
// For a real application, replace with actual API calls
class AuthService {
  async login(email: string, password: string): Promise<User> {
    try {
      // In a real app, this would be an API call
      // const response = await axios.post(`${API_URL}/auth/login`, { email, password });
      // return response.data.user;
      
      // Mock implementation
      await new Promise(resolve => setTimeout(resolve, 800)); // Simulate network delay
      
      // Mock authentication - replace with actual API call in production
      if (email === 'demo@example.com' && password === 'password') {
        const user = {
          id: '1',
          username: 'Demo User',
          email: 'demo@example.com'
        };
        
        // Store token - in a real app this would come from the server
        localStorage.setItem('token', 'mock-jwt-token');
        
        return user;
      } else {
        throw new Error('Invalid email or password');
      }
    } catch (error) {
      console.error('Login error:', error);
      throw error;
    }
  }
  
  async register(username: string, email: string, password: string): Promise<User> {
    try {
      // In a real app, this would be an API call
      // const response = await axios.post(`${API_URL}/auth/register`, { username, email, password });
      // return response.data.user;
      
      // Mock implementation
      await new Promise(resolve => setTimeout(resolve, 1000)); // Simulate network delay
      
      // Mock registration - replace with actual API call in production
      const user = {
        id: '1',
        username,
        email
      };
      
      // Store token - in a real app this would come from the server
      localStorage.setItem('token', 'mock-jwt-token');
      
      return user;
    } catch (error) {
      console.error('Registration error:', error);
      throw error;
    }
  }
  
  logout(): void {
    localStorage.removeItem('token');
    localStorage.removeItem('user');
  }
  
  async getCurrentUser(): Promise<User | null> {
    try {
      // In a real app, this would be an API call
      // const response = await axios.get(`${API_URL}/auth/user`);
      // return response.data.user;
      
      // Mock implementation
      const userString = localStorage.getItem('user');
      if (userString) {
        return JSON.parse(userString);
      }
      return null;
    } catch (error) {
      console.error('Get current user error:', error);
      return null;
    }
  }
  
  isAuthenticated(): boolean {
    return !!localStorage.getItem('token');
  }
}

export const authService = new AuthService(); 