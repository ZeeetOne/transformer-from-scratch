import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Home from './pages/Home';
import Applications from './pages/Applications';
import Mode1 from './pages/Mode1';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/applications" element={<Applications />} />
        <Route path="/applications/mode1" element={<Mode1 />} />
      </Routes>
    </Router>
  );
}

export default App;
