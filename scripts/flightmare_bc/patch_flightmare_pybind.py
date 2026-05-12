"""Patch Flightmare's pybind wrapper with a visual racing helper.

The upstream ``flightgym`` wheel only exposes ``QuadrotorEnv_v1``. That class is
useful for vectorized state/action dynamics, but it does not expose camera
frames, gates, or direct state control from Python. This patch adds
``VisualRacingEnv_v0`` to the same extension before the wheel is built.
"""
from __future__ import annotations

from pathlib import Path


VISUAL_RACING_BINDING = r'''

class VisualRacingEnv {
 public:
  VisualRacingEnv(const int scene_id = UnityScene::WAREHOUSE,
                  const int width = 224,
                  const int height = 224,
                  const Scalar fov = 90.0)
      : scene_id_(scene_id), width_(width), height_(height), fov_(fov) {
    const char* root = std::getenv("FLIGHTMARE_PATH");
    std::string cfg_path = root == nullptr
        ? std::string("/opt/flightmare/flightlib/configs/quadrotor_env.yaml")
        : std::string(root) + "/flightlib/configs/quadrotor_env.yaml";
    quad_ = std::make_shared<Quadrotor>(cfg_path);
    camera_ = std::make_shared<RGBCamera>();
    camera_->setWidth(width_);
    camera_->setHeight(height_);
    camera_->setFOV(fov_);
    Vector<3> B_r_BC(0.0, 0.0, 0.3);
    Matrix<3, 3> R_BC = Quaternion(1.0, 0.0, 0.0, 0.0).toRotationMatrix();
    camera_->setRelPose(B_r_BC, R_BC);
    camera_->setPostProcesscing(std::vector<bool>{false, false, false});
    quad_->addRGBCamera(camera_);
    bridge_ = std::make_shared<UnityBridge>();
    bridge_->addQuadrotor(quad_);
    reset(Vector<3>(0.0, 0.0, 2.0), Vector<4>(1.0, 0.0, 0.0, 0.0));
  }

  bool connectUnity() {
    unity_ready_ = bridge_->connectUnity(static_cast<SceneID>(scene_id_));
    return unity_ready_;
  }

  void disconnectUnity() {
    bridge_->disconnectUnity();
    unity_ready_ = false;
  }

  void reset(const Vector<3>& pos, const Vector<4>& quat_wxyz) {
    state_.setZero();
    state_.t = 0.0;
    state_.p = pos;
    state_.qx = quat_wxyz;
    quad_->reset(state_);
  }

  void setState(const Vector<3>& pos, const Vector<4>& quat_wxyz,
                const Vector<3>& vel, const Vector<3>& omega) {
    state_.setZero();
    state_.t = 0.0;
    state_.p = pos;
    state_.qx = quat_wxyz;
    state_.v = vel;
    state_.w = omega;
    quad_->setState(state_);
  }

  std::vector<Scalar> getState() {
    quad_->getState(&state_);
    return std::vector<Scalar>{
        state_.p.x(), state_.p.y(), state_.p.z(),
        state_.qx[0], state_.qx[1], state_.qx[2], state_.qx[3],
        state_.v.x(), state_.v.y(), state_.v.z(),
        state_.w.x(), state_.w.y(), state_.w.z()};
  }

  void stepMotor(const Vector<4>& thrusts_newton, const Scalar dt) {
    Command cmd(state_.t + dt, thrusts_newton);
    quad_->run(cmd, dt);
    quad_->getState(&state_);
  }

  void stepCtbr(const Scalar thrust_newton, const Vector<3>& body_rates,
                const Scalar dt) {
    Command cmd(state_.t + dt, thrust_newton / quad_->getMass(), body_rates);
    quad_->run(cmd, dt);
    quad_->getState(&state_);
  }

  void addGate(const std::string& object_id, const Vector<3>& pos,
               const Scalar yaw, const Vector<3>& size) {
    auto gate = std::make_shared<StaticGate>(object_id, "rpg_gate");
    gate->setPosition(pos);
    gate->setQuaternion(Quaternion(std::cos(0.5 * yaw), 0.0, 0.0,
                                   std::sin(0.5 * yaw)));
    gate->setSize(size);
    bridge_->addStaticObject(gate);
  }

  void addGateQuat(const std::string& object_id, const Vector<3>& pos,
                   const Vector<4>& quat_wxyz, const Vector<3>& size) {
    auto gate = std::make_shared<StaticGate>(object_id, "rpg_gate");
    gate->setPosition(pos);
    gate->setQuaternion(Quaternion(quat_wxyz[0], quat_wxyz[1],
                                   quat_wxyz[2], quat_wxyz[3]));
    gate->setSize(size);
    bridge_->addStaticObject(gate);
  }

  bool render() {
    if (!unity_ready_) return false;
    bridge_->getRender(frame_id_++);
    return bridge_->handleOutput();
  }

  py::array_t<uint8_t> getRGB() {
    cv::Mat bgr;
    if (!camera_->getRGBImage(bgr)) {
      return py::array_t<uint8_t>({height_, width_, 3});
    }
    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    py::array_t<uint8_t> out({rgb.rows, rgb.cols, rgb.channels()});
    std::memcpy(out.mutable_data(), rgb.data,
                static_cast<size_t>(rgb.rows * rgb.cols * rgb.channels()));
    return out;
  }

 private:
  int scene_id_;
  int width_;
  int height_;
  Scalar fov_;
  FrameID frame_id_{0};
  bool unity_ready_{false};
  std::shared_ptr<Quadrotor> quad_;
  std::shared_ptr<RGBCamera> camera_;
  std::shared_ptr<UnityBridge> bridge_;
  QuadState state_;
};
'''


VISUAL_RACING_PYBIND = r'''

  py::class_<VisualRacingEnv>(m, "VisualRacingEnv_v0")
    .def(py::init<const int, const int, const int, const Scalar>(),
         py::arg("scene_id") = 1,
         py::arg("width") = 224,
         py::arg("height") = 224,
         py::arg("fov") = 90.0)
    .def("connectUnity", &VisualRacingEnv::connectUnity)
    .def("disconnectUnity", &VisualRacingEnv::disconnectUnity)
    .def("reset", &VisualRacingEnv::reset)
    .def("setState", &VisualRacingEnv::setState)
    .def("getState", &VisualRacingEnv::getState)
    .def("stepMotor", &VisualRacingEnv::stepMotor)
    .def("stepCtbr", &VisualRacingEnv::stepCtbr)
    .def("addGate", &VisualRacingEnv::addGate,
         py::arg("object_id"), py::arg("pos"), py::arg("yaw"),
         py::arg("size"))
    .def("addGateQuat", &VisualRacingEnv::addGateQuat,
         py::arg("object_id"), py::arg("pos"), py::arg("quat_wxyz"),
         py::arg("size"))
    .def("render", &VisualRacingEnv::render)
    .def("getRGB", &VisualRacingEnv::getRGB);
'''


def main() -> None:
    wrapper = Path("/opt/flightmare/flightlib/src/wrapper/pybind_wrapper.cpp")
    text = wrapper.read_text()
    if "VisualRacingEnv_v0" in text:
        patched = text
        patched = patched.replace(
            "reset(Vector<3>(0.0, 0.0, 2.0), Quaternion(1.0, 0.0, 0.0, 0.0));",
            "reset(Vector<3>(0.0, 0.0, 2.0), Vector<4>(1.0, 0.0, 0.0, 0.0));",
        )
        patched = patched.replace(
            """  void reset(const Vector<3>& pos, const Quaternion& quat) {
    state_.setZero();
    state_.t = 0.0;
    state_.p = pos;
    state_.qx << quat.w(), quat.x(), quat.y(), quat.z();
    quad_->reset(state_);
  }""",
            """  void reset(const Vector<3>& pos, const Vector<4>& quat_wxyz) {
    state_.setZero();
    state_.t = 0.0;
    state_.p = pos;
    state_.qx = quat_wxyz;
    quad_->reset(state_);
  }""",
        )
        if patched != text:
            wrapper.write_text(patched)
        if "addGateQuat" not in patched:
            patched = patched.replace(
                """  bool render() {
    if (!unity_ready_) return false;
    bridge_->getRender(frame_id_++);
    return bridge_->handleOutput();
  }""",
                """  void addGateQuat(const std::string& object_id, const Vector<3>& pos,
                   const Vector<4>& quat_wxyz, const Vector<3>& size) {
    auto gate = std::make_shared<StaticGate>(object_id, "rpg_gate");
    gate->setPosition(pos);
    gate->setQuaternion(Quaternion(quat_wxyz[0], quat_wxyz[1],
                                   quat_wxyz[2], quat_wxyz[3]));
    gate->setSize(size);
    bridge_->addStaticObject(gate);
  }

  bool render() {
    if (!unity_ready_) return false;
    bridge_->getRender(frame_id_++);
    return bridge_->handleOutput();
  }""",
            )
            patched = patched.replace(
                """.def("addGate", &VisualRacingEnv::addGate,
         py::arg("object_id"), py::arg("pos"), py::arg("yaw"),
         py::arg("size"))""",
                """.def("addGate", &VisualRacingEnv::addGate,
         py::arg("object_id"), py::arg("pos"), py::arg("yaw"),
         py::arg("size"))
    .def("addGateQuat", &VisualRacingEnv::addGateQuat,
         py::arg("object_id"), py::arg("pos"), py::arg("quat_wxyz"),
         py::arg("size"))""",
            )
            wrapper.write_text(patched)
        return

    include_anchor = '#include "flightlib/envs/vec_env.hpp"\n'
    includes = (
        include_anchor
        + '#include "flightlib/objects/static_gate.hpp"\n'
        + '#include "flightlib/sensors/rgb_camera.hpp"\n'
        + "#include <opencv2/imgproc/imgproc.hpp>\n"
        + "#include <cstring>\n"
    )
    text = text.replace(include_anchor, includes)
    text = text.replace("namespace py = pybind11;\nusing namespace flightlib;\n",
                        "namespace py = pybind11;\nusing namespace flightlib;\n" + VISUAL_RACING_BINDING)
    text = text.replace('  py::class_<TestEnv<QuadrotorEnv>>(m, "TestEnv_v0")',
                        VISUAL_RACING_PYBIND + '\n  py::class_<TestEnv<QuadrotorEnv>>(m, "TestEnv_v0")')
    wrapper.write_text(text)


if __name__ == "__main__":
    main()
