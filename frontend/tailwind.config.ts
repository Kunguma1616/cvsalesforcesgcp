/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        primary: {
          50: '#F7F9FD',
          100: '#DEE8F7',
          200: '#7099DB',
          300: '#7099DB',
          400: '#4A72B8',
          500: '#27549D',
          600: '#1F4180',
          700: '#17325E',
          800: '#0F2341',
          900: '#081424',
        },
        secondary: {
          50: '#FEF5EC',
          100: '#FCE9D4',
          200: '#F7C182',
          300: '#F29630',
          400: '#D17D0A',
          500: '#A35C0A',
          600: '#8B4B08',
          700: '#5C3205',
          800: '#3B2003',
          900: '#1D1001',
        },
        accent: {
          50: '#FFFDE7',
          100: '#FFF59D',
          200: '#FFF59D',
          300: '#F1FF24',
          400: '#E0E81C',
          500: '#CFD714',
          600: '#ADB50F',
          700: '#8B930A',
          800: '#697105',
          900: '#3D4802',
        },
      },
      fontFamily: {
        sans: ['Mont'],
      },
      boxShadow: {
        soft: '0 2px 8px rgba(0, 0, 0, 0.1)',
        md: '0 4px 12px rgba(0, 0, 0, 0.15)',
        lg: '0 10px 25px rgba(0, 0, 0, 0.2)',
      },
      borderRadius: {
        xl: '16px',
        '2xl': '24px',
      },
    },
  },
  plugins: [],
};
