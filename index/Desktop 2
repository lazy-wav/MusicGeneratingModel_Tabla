import React from "react";

import { useNavigate } from "react-router-dom";

import { Button, Img, Line, Text } from "components";

const DesktopTwoPage = () => {
  const navigate = useNavigate();

  return (
    <>
      <div className="bg-blue_gray-900 flex flex-col font-inter items-center justify-start mx-auto pb-8 w-full">
        <header
          className="bg-cover bg-no-repeat flex md:flex-col md:gap-5 h-[122px] items-center justify-center md:px-5 w-full"
          style={{ backgroundImage: "url('images/img_group11.svg')" }}
        >
          <div className="flex md:flex-1 flex-col items-center justify-start mb-[11px] mr-10 md:mt-0 mt-[17px] w-[98%] md:w-full">
            <div className="flex md:flex-col flex-row md:gap-5 items-end justify-start w-full">
              <Img
                className="md:flex-1 h-[87px] sm:h-auto mb-1.5 object-cover w-[12%] md:w-full"
                src="images/img_sihlogoremovebgpreview.png"
                alt="sihlogoremovebg"
              />
              <div
                className="common-pointer flex md:flex-1 flex-col gap-2.5 items-center justify-start md:ml-[0] ml-[863px] md:mt-0 mt-[9px] w-[5%] md:w-full"
                onClick={() => navigate("/")}
              >
                <Img
                  className="h-[50px] md:h-auto rounded-[50%] w-[50px]"
                  src="images/img_home1.png"
                  alt="homeOne"
                />
                <Text
                  className="text-black-900 text-lg tracking-[1.80px]"
                  size="txtInterExtraBold18"
                >
                  Home
                </Text>
              </div>
              <div className="flex md:flex-1 flex-col gap-3 items-center justify-start ml-5 md:ml-[0] md:mt-0 mt-3 w-[9%] md:w-full">
                <Img
                  className="h-[47px] md:h-auto object-cover w-[46px] sm:w-full"
                  src="images/img_language1.png"
                  alt="languageOne"
                />
                <Text
                  className="text-black-900 text-lg tracking-[1.80px]"
                  size="txtInterExtraBold18"
                >
                  Languages
                </Text>
              </div>
              <div className="flex md:flex-1 flex-col gap-2.5 items-center justify-start ml-5 md:ml-[0] md:mt-0 mt-[9px] w-[7%] md:w-full">
                <Img
                  className="h-[50px] md:h-auto object-cover w-[42%] sm:w-full"
                  src="images/img_telephone1.png"
                  alt="telephoneOne"
                />
                <Text
                  className="text-black-900 text-lg tracking-[1.80px]"
                  size="txtInterExtraBold18"
                >
                  Contact
                </Text>
              </div>
              <div
                className="common-pointer flex md:flex-1 flex-col gap-[11px] items-center justify-start ml-5 md:ml-[0] md:mt-0 mt-[17px] w-[5%] md:w-full"
                onClick={() => navigate("/login")}
              >
                <Img
                  className="h-[42px] md:h-auto object-cover w-[42px]"
                  src="images/img_userinterface1.png"
                  alt="userinterfaceOne"
                />
                <a
                  href="javascript:"
                  className="text-black-900 text-lg tracking-[1.80px]"
                >
                  <Text size="txtInterExtraBold18">Log in</Text>
                </a>
              </div>
            </div>
          </div>
          <Text
            className="mb-[39px] md:ml-[0] ml-[127px] mr-[1112px] md:mt-0 mt-[43px] md:text-3xl sm:text-[28px] text-[32px] text-cyan-700 tracking-[3.20px]"
            size="txtInterExtraBold32"
          >
            RescueNet
          </Text>
        </header>
        <div className="bg-blue_gray-900 flex md:flex-col flex-row md:gap-5 items-center justify-evenly overflow-auto w-full">
          <div className="h-[107px] md:px-5 relative w-1/5 md:w-full">
            <div className="absolute bg-blue_gray-900 h-[107px] inset-y-[0] my-auto right-[0] w-1/2"></div>
            <Text
              className="common-pointer absolute h-max inset-y-[0] my-auto right-[0] text-center text-lg text-white-A700 tracking-[1.80px]"
              size="txtInterMedium18"
              onClick={() => navigate("/weather")}
            >
              <>
                Weather
                <br /> Alert
              </>
            </Text>
            <Text
              className="absolute h-max inset-y-[0] left-[0] my-auto text-center text-lg text-white-A700 tracking-[1.80px]"
              size="txtInterMedium18"
            >
              <>
                Emergency
                <br />
                Alert
              </>
            </Text>
          </div>
          <div className="h-[107px] md:px-5 relative w-[10%] md:w-full">
            <div className="absolute bg-blue_gray-900 h-[107px] inset-[0] justify-center m-auto w-full"></div>
            <Text
              className="common-pointer absolute h-max inset-[0] justify-center m-auto text-center text-lg text-white-A700 tracking-[1.80px]"
              size="txtInterMedium18"
              onClick={() => navigate("/info")}
            >
              <>
                Safety
                <br />
                Information
              </>
            </Text>
          </div>
          <div className="bg-blue_gray-900 flex flex-col items-center justify-start p-[9px] md:px-5">
            <Text
              className="my-3.5 text-center text-lg text-white-A700 tracking-[1.80px]"
              size="txtInterMedium18"
            >
              <>
                Volunteer
                <br />
                Registration
              </>
            </Text>
          </div>
          <Button
            className="cursor-pointer font-medium leading-[normal] min-w-[143px] text-center text-lg tracking-[1.80px]"
            shape="square"
          >
            Response
          </Button>
          <Button
            className="cursor-pointer font-medium leading-[normal] min-w-[143px] text-center text-lg tracking-[1.80px]"
            shape="square"
          >
            Resources
          </Button>
          <div className="h-[107px] md:px-5 relative w-[10%] md:w-full">
            <div className="absolute bg-blue_gray-900 h-[107px] inset-[0] justify-center m-auto w-full"></div>
            <Text
              className="absolute h-max inset-[0] justify-center m-auto text-center text-lg text-white-A700 tracking-[1.80px]"
              size="txtInterMedium18"
            >
              <>
                Donate/
                <br />
                contribute
              </>
            </Text>
          </div>
          <div className="h-[107px] sm:h-[234px] md:px-5 relative w-[30%] md:w-full">
            <div className="absolute flex flex-col h-full inset-[0] items-center justify-center m-auto w-full">
              <div className="flex sm:flex-col flex-row sm:gap-5 items-center justify-evenly w-full">
                <div className="bg-blue_gray-900 h-[107px] w-[34%]"></div>
                <div className="flex sm:flex-1 flex-row items-center justify-evenly w-[67%] sm:w-full">
                  <div className="h-[107px] relative w-1/2">
                    <div className="absolute bg-blue_gray-900 h-[107px] inset-[0] justify-center m-auto w-full"></div>
                    <Text
                      className="absolute h-max inset-[0] justify-center m-auto text-center text-lg text-white-A700 tracking-[1.80px]"
                      size="txtInterMedium18"
                    >
                      <>
                        Emergency
                        <br />
                        Contacts
                      </>
                    </Text>
                  </div>
                  <div className="bg-blue_gray-900 h-[107px] w-1/2"></div>
                </div>
              </div>
            </div>
            <Text
              className="absolute h-max inset-y-[0] left-[0] my-auto text-center text-lg text-white-A700 tracking-[1.80px]"
              size="txtInterMedium18"
            >
              <>
                Report
                <br />
                Incident
              </>
            </Text>
            <Text
              className="absolute h-max inset-y-[0] my-auto right-[0] text-center text-lg text-white-A700 tracking-[1.80px]"
              size="txtInterMedium18"
            >
              <>
                Find nearby
                <br />
                Agencies
              </>
            </Text>
          </div>
        </div>
        <Line className="bg-white-A700 h-px max-w-[1418px] mx-auto w-full" />
        <div className="backdrop-opacity-[0.5] bg-cyan-50_01 blur-[4.00px] flex flex-col font-inder gap-11 items-center justify-start mt-[35px] p-[13px] md:px-5 rounded-[30px] w-[24%] md:w-full">
          <Text
            className="text-black-900 text-xl tracking-[2.00px]"
            size="txtInderRegular20"
          >
            Select any one{" "}
          </Text>
          <div className="flex flex-col items-center justify-start mb-[67px] w-[85%] md:w-full">
            <div className="h-[74px] relative w-full">
              <div className="bg-blue_gray-900 h-[74px] m-auto rounded-[133px] shadow-bs w-full"></div>
              <Text
                className="absolute bottom-[16%] left-1/4 text-3xl sm:text-[26px] md:text-[28px] text-gray-50_01 tracking-[3.00px]"
                size="txtInderRegular30"
              >
                1.Hindi
              </Text>
            </div>
            <div className="md:h-[125px] h-[74px] mt-[51px] relative w-full">
              <div className="bg-blue_gray-900 h-[74px] m-auto rounded-[133px] shadow-bs w-full"></div>
              <Text
                className="absolute bottom-[20%] left-[17%] text-3xl sm:text-[26px] md:text-[28px] text-gray-50_02 tracking-[3.00px]"
                size="txtInderRegular30Gray5002"
              >
                2.English
              </Text>
            </div>
            <div className="md:h-[120px] h-[74px] mt-[46px] relative w-full">
              <div className="bg-blue_gray-900 h-[74px] m-auto rounded-[133px] shadow-bs w-full"></div>
              <Text
                className="absolute bottom-[20%] inset-x-[0] mx-auto text-3xl sm:text-[26px] md:text-[28px] text-white-A700 tracking-[3.00px] w-max"
                size="txtInderRegular30WhiteA700"
              >
                3.Malayalam
              </Text>
            </div>
            <div className="md:h-[123px] h-[74px] mt-[49px] relative w-full">
              <div className="bg-blue_gray-900 h-[74px] m-auto rounded-[133px] shadow-bs w-full"></div>
              <Text
                className="absolute h-max inset-y-[0] left-[19%] my-auto text-3xl sm:text-[26px] md:text-[28px] text-gray-50_03 tracking-[3.00px]"
                size="txtInderRegular30Gray5003"
              >
                4.Gujrati{" "}
              </Text>
            </div>
            <div className="md:h-[120px] h-[74px] mt-[46px] relative w-full">
              <div className="bg-blue_gray-900 h-[74px] m-auto rounded-[133px] shadow-bs w-full"></div>
              <Text
                className="absolute bottom-[19%] inset-x-[0] mx-auto text-3xl sm:text-[26px] md:text-[28px] text-white-A700 tracking-[3.00px] w-max"
                size="txtInderRegular30WhiteA700"
              >
                5.Punjabi
              </Text>
            </div>
          </div>
        </div>
      </div>
    </>
  );
};

export default DesktopTwoPage;
