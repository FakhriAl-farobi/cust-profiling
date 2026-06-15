/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  theme: {
    extend: {
      colors: {
        corporate: {
          50: '#f5f9ff',
          100: '#e7f0ff',
          200: '#c8dcff',
          300: '#9bbfff',
          400: '#6598f5',
          500: '#3775df',
          600: '#205abb',
          700: '#194994',
          800: '#153d78',
          900: '#102f5f',
          950: '#071a38'
        },
        ink: '#122033',
        muted: '#66758a',
        line: '#d9e3f2'
      },
      fontFamily: {
        sans: ['Inter', 'Segoe UI', 'system-ui', 'sans-serif']
      },
      boxShadow: {
        soft: '0 18px 45px rgba(25, 73, 148, 0.10)'
      }
    }
  },
  plugins: []
}
