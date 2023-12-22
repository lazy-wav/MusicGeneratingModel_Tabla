import React from "react";

import { useNavigate } from "react-router-dom";

import { Button, Img, Text } from "components";

const InfoPage = () => {
  const navigate = useNavigate();

  return (
    <>
      <div className="bg-cyan-50 flex flex-col items-center justify-start mx-auto pb-[42px] w-full">
        <div className="flex flex-col items-center justify-start w-full">
          <header className="bg-cyan-50 flex md:flex-col flex-row font-inter md:gap-5 items-center justify-center md:px-5 w-full">
            <div className="flex md:mt-0 my-[17px] relative w-[23%] md:w-full">
              <Img
                className="h-[87px] my-auto object-cover w-[48%]"
                src="images/img_sihlogoremovebgpreview.png"
                alt="sihlogoremovebg"
              />
              <Text
                className="mb-[22px] ml-[-29.8px] mt-auto md:text-3xl sm:text-[28px] text-[32px] text-cyan-700 tracking-[3.20px] z-[1]"
                size="txtInterExtraBold32"
              >
                RescueNet
              </Text>
            </div>
            <div className="flex md:flex-1 flex-col gap-2.5 items-center justify-start mb-[13px] md:ml-[0] ml-[692px] md:mt-0 mt-[26px] w-[5%] md:w-full">
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
            <div className="flex md:flex-1 flex-col gap-3 items-center justify-start mb-[11px] ml-5 md:ml-[0] md:mt-0 mt-[29px] w-[8%] md:w-full">
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
            <div className="flex md:flex-1 flex-col gap-2.5 items-center justify-start mb-[13px] ml-5 md:ml-[0] md:mt-0 mt-[26px] w-[6%] md:w-full">
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
            <div className="flex md:flex-1 flex-col gap-[11px] items-center justify-start mb-3 ml-5 md:ml-[0] mr-10 md:mt-0 mt-[34px] w-[5%] md:w-full">
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
          </header>
          <div className="flex md:flex-col flex-row font-inter md:gap-5 items-center justify-evenly overflow-auto w-full">
            <div className="h-[107px] md:px-5 relative w-1/5 md:w-full">
              <div className="flex flex-row h-full items-center justify-evenly m-auto w-full">
                <div className="bg-blue_gray-900 h-[107px] w-1/2"></div>
                <div className="h-[107px] relative w-1/2">
                  <div className="absolute bg-blue_gray-900 h-[107px] inset-[0] justify-center m-auto w-full"></div>
                  <Text
                    className="common-pointer absolute h-max inset-[0] justify-center m-auto text-center text-lg text-white-A700 tracking-[1.80px]"
                    size="txtInterMedium18"
                    onClick={() => navigate("/weather")}
                  >
                    <>
                      Weather
                      <br /> Alert
                    </>
                  </Text>
                </div>
              </div>
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
                className="absolute h-max inset-[0] justify-center m-auto text-center text-lg text-white-A700 tracking-[1.80px]"
                size="txtInterMedium18"
              >
                <>
                  Safety
                  <br />
                  Information
                </>
              </Text>
            </div>
            <Button
              className="cursor-pointer font-medium leading-[normal] text-center text-lg tracking-[1.80px]"
              shape="square"
              size="md"
            >
              <>
                Volunteer
                <br />
                Registration
              </>
            </Button>
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
          <Text
            className="mt-[17px] md:text-5xl text-7xl text-black-900 text-center"
            size="txtIMFELLFrenchCanonSC72"
          >
            SAFETY INFORMATION
          </Text>
          <div className="font-inconsolata md:gap-5 gap-[55px] grid sm:grid-cols-1 md:grid-cols-2 grid-cols-3 justify-center max-w-[1253px] min-h-[auto] mt-[5px] mx-auto md:px-5 w-full">
            <div className="bg-blue_gray-900 flex flex-1 flex-col justify-start p-[15px] w-full">
              <Img
                className="h-[91px] md:h-auto md:ml-[0] ml-[108px] object-cover w-[28%] sm:w-full"
                src="images/img_earthquake1.png"
                alt="earthquakeOne"
              />
              <Text
                className="md:ml-[0] ml-[22px] mr-[31px] my-1 sm:text-[39.79px] md:text-[45.79px] text-[53.79px] text-white-A700"
                size="txtInconsolataRegular5379"
              >
                EARTHQUAKES
              </Text>
            </div>
            <div className="bg-blue_gray-900 flex flex-1 flex-col items-center justify-start p-[15px] w-full">
              <Img
                className="h-[91px] md:h-auto object-cover w-[28%] sm:w-full"
                src="images/img_landslide2.png"
                alt="landslideTwo"
              />
              <Text
                className="mb-[9px] sm:text-[39.79px] md:text-[45.79px] text-[53.79px] text-white-A700"
                size="txtInconsolataRegular5379"
              >
                LANDSLIDES
              </Text>
            </div>
            <div className="bg-blue_gray-900 flex flex-1 flex-col items-center justify-start p-[15px] w-full">
              <Img
                className="h-[91px] md:h-auto object-cover w-[28%] sm:w-full"
                src="images/img_floodedhouse2.png"
                alt="floodedhouseTwo"
              />
              <Text
                className="mb-[9px] sm:text-[39.79px] md:text-[45.79px] text-[53.79px] text-white-A700"
                size="txtInconsolataRegular5379"
              >
                FLOODS
              </Text>
            </div>
            <div className="bg-blue_gray-900 flex flex-1 flex-col items-center justify-end p-[15px] w-full">
              <Img
                className="h-[91px] md:h-auto mt-2.5 object-cover w-[28%] sm:w-full"
                src="images/img_image7.png"
                alt="imageSeven"
              />
              <Text
                className="sm:text-[39.79px] md:text-[45.79px] text-[53.79px] text-white-A700"
                size="txtInconsolataRegular5379"
              >
                TSUNAMIS
              </Text>
            </div>
            <div className="bg-blue_gray-900 flex flex-1 flex-col items-center justify-start p-[15px] w-full">
              <Img
                className="h-[91px] md:h-auto object-cover w-[28%] sm:w-full"
                src="images/img_image8.png"
                alt="imageEight"
              />
              <Text
                className="mb-[9px] sm:text-[39.79px] md:text-[45.79px] text-[53.79px] text-white-A700"
                size="txtInconsolataRegular5379"
              >
                LIGHTNING
              </Text>
            </div>
            <div className="bg-blue_gray-900 flex flex-1 flex-col items-center justify-start p-[15px] w-full">
              <Img
                className="h-[91px] md:h-auto object-cover w-[28%] sm:w-full"
                src="images/img_image9.png"
                alt="imageNine"
              />
              <Text
                className="mb-2 sm:text-[39.79px] md:text-[45.79px] text-[53.79px] text-white-A700"
                size="txtInconsolataRegular5379"
              >
                TORNADOES
              </Text>
            </div>
            <div className="bg-blue_gray-900 flex flex-1 flex-col items-center justify-start p-4 w-full">
              <Img
                className="h-[91px] md:h-auto object-cover w-[28%] sm:w-full"
                src="images/img_image4.png"
                alt="imageFour"
              />
              <Text
                className="mb-[7px] sm:text-[39.79px] md:text-[45.79px] text-[53.79px] text-white-A700"
                size="txtInconsolataRegular5379"
              >
                WILDFIRES
              </Text>
            </div>
            <div className="bg-blue_gray-900 flex flex-1 flex-col items-center justify-start p-4 w-full">
              <Img
                className="h-[91px] md:h-auto object-cover w-[28%] sm:w-full"
                src="images/img_image5.png"
                alt="imageFive"
              />
              <Text
                className="mb-[7px] sm:text-[39.79px] md:text-[45.79px] text-[53.79px] text-white-A700"
                size="txtInconsolataRegular5379"
              >
                HURRICANES
              </Text>
            </div>
            <div className="bg-blue_gray-900 flex flex-1 flex-col items-center justify-start p-4 w-full">
              <Img
                className="h-[91px] md:h-auto object-cover w-[28%] sm:w-full"
                src="images/img_image6.png"
                alt="imageSix"
              />
              <Text
                className="mb-2 sm:text-[39.79px] md:text-[45.79px] text-[53.79px] text-white-A700"
                size="txtInconsolataRegular5379"
              >
                EXTREME HEAT
              </Text>
            </div>
          </div>
        </div>
      </div>
    </>
  );
};

export default InfoPage;
