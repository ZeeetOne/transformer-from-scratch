import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Home from '../pages/Home';
import Applications from '../pages/Applications';
import { Mode1Page } from '../features/mode1-next-word';

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/applications" element={<Applications />} />
        <Route path="/applications/mode1" element={<Mode1Page />} />
      </Routes>
    </Router>
  );
}

export default App;
